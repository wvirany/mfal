import torch
from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.api.reward import Reward
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateEarlyTerminal
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.objectives.rgfn_trajectory_filter import RGFNTrajectoryFilter
from rgfn.gfns.reaction_gfn.policies.action_embeddings import FragmentFingerprintEmbedding
from rgfn.gfns.reaction_gfn.policies.reaction_backward_policy import ReactionBackwardPolicy
from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import ReactionForwardPolicy
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy
from rgfn.gfns.reaction_gfn.reaction_env import ReactionEnv
from rgfn.shared.objectives.trajectory_balance_objective import TrajectoryBalanceObjective
from rgfn.shared.replay_buffers.reward_prioritized_replay_buffer import (
    RewardPrioritizedReplayBuffer,
)
from rgfn.shared.samplers.random_sampler import RandomSampler
from rgfn.trainer.logger.dummy_logger import DummyLogger
from rgfn.trainer.optimizers.trajectory_balance_optimizer import TrajectoryBalanceOptimizer
from rgfn.trainer.trainer import Trainer

from molbo.acqf_optimizer import AcqfOptimizer
from molbo.utils import smiles_to_morgan_fp


class AcquisitionProxy(ProxyBase):
    """
    Wraps an acquisition function as an RGFN proxy.
    Updated each BO iteration via update()
    """

    def __init__(self):
        super().__init__()
        self.acq_func = None

    def update(self, acq_func):
        self.acq_func = acq_func

    def compute_proxy_output(self, states):
        if self.acq_func is None:
            raise RuntimeError(
                "AcquisitionProxy.acq_func is not set. Call update() before training."
            )

        valid_mask = [not isinstance(s, ReactionStateEarlyTerminal) for s in states]
        valid_states = [s for s, v in zip(states, valid_mask) if v]

        smiles = [s.molecule.smiles for s in valid_states]
        X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])  # (n, 2048)
        X = X.reshape(-1, 1, X.shape[-1])  # (n, 1, 2048) for BoTorch

        with torch.no_grad():
            acq_values = self.acq_func(X).float()  # (n,)

        result = torch.zeros(len(states), dtype=torch.float32)
        result[torch.tensor(valid_mask)] = acq_values

        return ProxyOutput(value=result, components=None)

    @property
    def is_non_negative(self):
        return True

    @property
    def higher_is_better(self):
        return True


class RGFN(AcqfOptimizer):
    """
    AcqfOptimizer that uses RGFN to generate candidates.
    Internally manages all RGFN components with default settings.

    Args:
        reaction_path: Path to reaction templates file
        fragment_path: Path to fragments file
        M: Number of candidates to sample from RGFN per BO iteration
        q: Number of candidates to return (top-q by acquisition value)
        n_iterations: Number of RGFN training steps per BO iteration
        run_dir: Directory for RGFN temporary files
    """

    def __init__(
        self,
        reaction_path: str,
        fragment_path: str,
        M: int,
        q: int,
        n_iterations: int,
        run_dir: str = "./tmp_rgfn",
    ):
        self.M = M
        self.q = q
        self.n_iterations = n_iterations

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # RGFN prefers torch.float32, molbo prefers torch.float64
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        data_factory = ReactionDataFactory(
            reaction_path=reaction_path,
            fragment_path=fragment_path,
            cost_path=None,
            yield_path=None,
        )
        env = ReactionEnv(data_factory=data_factory, max_num_reactions=3)

        self.proxy = AcquisitionProxy()
        reward = Reward(proxy=self.proxy, reward_boosting="exponential", beta=32, min_reward=1e-8)

        action_embedding_fn = FragmentFingerprintEmbedding(
            data_factory=data_factory,
            fingerprint_list=["maccs"],
            dynamic_library=None,
            hidden_dim=64,
        )
        forward_policy = ReactionForwardPolicy(
            data_factory=data_factory,
            hidden_dim=64,
            action_embedding_fn=action_embedding_fn,
        )
        backward_policy = ReactionBackwardPolicy(
            data_factory=data_factory,
            hidden_dim=64,
        )

        self.forward_sampler = RandomSampler(env=env, policy=forward_policy, reward=reward)
        backward_sampler = RandomSampler(env=env.reversed(), policy=backward_policy, reward=reward)
        replay_buffer = RewardPrioritizedReplayBuffer(
            sampler=backward_sampler,
            max_size=int(1e6),
            temperature=32,
        )

        objective = TrajectoryBalanceObjective(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            trajectory_filter=RGFNTrajectoryFilter(),
            z_dim=16,
        )
        optimizer = TrajectoryBalanceOptimizer(
            cls_name="Adam",
            lr=0.001,
            logZ_multiplier=100.0,
        )

        self.trainer = Trainer(
            run_dir=run_dir,
            logger=DummyLogger(),
            train_forward_sampler=self.forward_sampler,
            train_replay_buffer=replay_buffer,
            train_forward_n_trajectories=64,
            train_backward_n_trajectories=0,
            train_replay_n_trajectories=32,
            objective=objective,
            optimizer=optimizer,
            n_iterations=n_iterations,
            path_cost_proxy=PathCostProxy(data_factory=data_factory, yield_value=0.75),
            checkpoint_mode="none",
            device=device,
        )

        torch.set_default_dtype(prev_dtype)

    def optimize(self, acq_func, candidates=None):
        # RGFN prefers torch.float32, molbo prefers torch.float64
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        # Update proxy and advance training window
        self.proxy.update(acq_func)
        self.trainer.n_iterations = self.trainer.start_iteration + self.n_iterations
        self.trainer.train()
        self.trainer.start_iteration = self.trainer.n_iterations

        # Sample M candidates from RGFN
        trajectories = self.forward_sampler.sample_trajectories_batch(
            n_total_trajectories=self.M, batch_size=self.M
        )
        terminal_states = trajectories.get_last_states_flat()
        smiles = [s.molecule.smiles for s in terminal_states]

        torch.set_default_dtype(prev_dtype)

        # Convert to fingerprints
        X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])  # (M, 2048)

        # Evaluate acquisition function and select top-q
        with torch.no_grad():
            acq_values = acq_func(X.reshape(-1, 1, X.shape[-1]))  # (M, q, d) --> (M,)

        top_q = acq_values.topk(self.q)
        new_X = X[top_q.indices]
        acq_val = top_q.values

        return new_X, acq_val, None
