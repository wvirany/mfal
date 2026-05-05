from pathlib import Path

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
from rgfn.shared.policies.uniform_policy import UniformPolicy
from rgfn.shared.replay_buffers.reward_prioritized_replay_buffer import (
    RewardPrioritizedReplayBuffer,
)
from rgfn.shared.samplers.random_sampler import RandomSampler
from rgfn.trainer.logger.dummy_logger import DummyLogger
from rgfn.trainer.optimizers.trajectory_balance_optimizer import TrajectoryBalanceOptimizer
from rgfn.trainer.trainer import Trainer

from molbo.acqf_optimizer import AcqfOptimizer, Initialization, OptimizationResult
from molbo.utils import smiles_to_morgan_fp

_RGFN_DATA = Path(__file__).parent.parent.parent / "data" / "rgfn"


def _compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """JS divergence between two probability distributions."""
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log((p + 1e-10) / (m + 1e-10))).sum()
    kl_qm = (q * torch.log((q + 1e-10) / (m + 1e-10))).sum()
    return (0.5 * (kl_pm + kl_qm)).item()


class AcquisitionProxy(ProxyBase):
    """
    Wraps an acquisition function as an RGFN proxy.
    Updated each BO iteration via update()
    """

    def __init__(self):
        super().__init__()
        self.acq_func = None
        self.device = "cpu"

    def update(self, acq_func):
        self.acq_func = acq_func

    def compute_proxy_output(self, states):
        if self.acq_func is None:
            raise RuntimeError(
                "AcquisitionProxy.acq_func is not set. Call update() before training."
            )

        valid_mask = [not isinstance(s, ReactionStateEarlyTerminal) for s in states]
        valid_states = [s for s, v in zip(states, valid_mask) if v]

        if not valid_states:
            return ProxyOutput(value=torch.zeros(len(states), dtype=torch.float32), components=None)

        smiles = [s.molecule.smiles for s in valid_states]
        X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])  # (n, 2048)
        X = X.reshape(-1, 1, X.shape[-1]).to(self.device)  # (n, 1, 2048) for BoTorch

        with torch.no_grad():
            acq_values = self.acq_func(X).float()  # (n,)

        if torch.isnan(acq_values).any():
            print("Warning: NaN acquisition values detected, replacing with zeros")
            acq_values = torch.nan_to_num(acq_values, nan=0.0)

        result = torch.zeros(len(states), dtype=torch.float32, device=self.device)
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
        reaction_path: str = str(_RGFN_DATA / "templates_30k.txt"),
        fragment_path: str = str(_RGFN_DATA / "fragments_30k.txt"),
        q: int = 100,
        n_iterations: int = 250,
        train_forward_n_trajectories: int = 64,
        train_replay_n_trajectories: int = 32,
        run_dir: str = "./tmp_rgfn",
    ):
        self.q = q
        self.n_iterations = n_iterations

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

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
        self.proxy.device = device
        reward = Reward(proxy=self.proxy, min_reward=1e-8)

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
            train_forward_n_trajectories=train_forward_n_trajectories,
            train_backward_n_trajectories=0,
            train_replay_n_trajectories=train_replay_n_trajectories,
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

        # Sample q candidates from RGFN
        seen_smiles = set()
        terminal_states = []

        while len(terminal_states) < self.q:
            trajectories = self.forward_sampler.sample_trajectories_batch(
                n_total_trajectories=self.q, batch_size=self.q
            )
            for s in trajectories.get_last_states_flat():
                if (
                    not isinstance(s, ReactionStateEarlyTerminal)
                    and s.molecule.smiles not in seen_smiles
                ):
                    seen_smiles.add(s.molecule.smiles)
                    terminal_states.append(s)
                if len(terminal_states) == self.q:
                    break

        smiles = [s.molecule.smiles for s in terminal_states]

        torch.set_default_dtype(prev_dtype)

        # Convert to fingerprints
        X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles]).to(self.device)  # (q, 2048)

        # Evaluate acquisition function and select top-q
        with torch.no_grad():
            acq_values = acq_func(X.reshape(-1, 1, X.shape[-1]))  # (q, q, d) --> (q,)

        return OptimizationResult(new_X=X, acq_val=acq_values, smiles=smiles)

    def sample_init(self, oracle, n_init: int) -> Initialization:
        uniform_sampler = RandomSampler(
            env=self.forward_sampler.env,
            policy=UniformPolicy(),
            reward=None,
        )

        sampled_states = []
        sampled_smiles = set()
        while len(sampled_states) < n_init:
            trajectories = uniform_sampler.sample_trajectories_batch(
                n_total_trajectories=n_init - len(sampled_states),
                batch_size=n_init - len(sampled_states),
            )
            terminal_states = trajectories.get_last_states_flat()
            for state in terminal_states:
                if (
                    not isinstance(state, ReactionStateEarlyTerminal)
                    and state.molecule.smiles not in sampled_smiles
                ):
                    sampled_states.append(state)
                    sampled_smiles.add(state.molecule.smiles)

        smiles = [s.molecule.smiles for s in sampled_states]
        train_X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])
        train_y = oracle(smiles)

        return Initialization(train_X=train_X, train_y=train_y, smiles=smiles)


class RGFNPoolSampler(RGFN):
    """RGFN with a fully enumerated state space."""

    def __init__(
        self,
        *args,
        max_batch_size: int = 1024,
        compute_js_divergence: bool = True,
        M_eval: int = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_batch_size = max_batch_size
        self.compute_js_divergence = compute_js_divergence
        self.M_eval = M_eval

    def optimize(self, acq_func, candidates: torch.Tensor):
        # Build local hash map: bytes -> local index in unobserved pool
        pool_map = {candidates[i].cpu().numpy().tobytes(): i for i in range(len(candidates))}

        # Train GFN (dtype management handled by parent)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.proxy.update(acq_func)
        self.trainer.n_iterations = self.trainer.start_iteration + self.n_iterations
        self.trainer.train()
        self.trainer.start_iteration = self.trainer.n_iterations
        torch.set_default_dtype(prev_dtype)

        # Sampling loop
        collected_X = []  # fingerprints, float64
        collected_idx = []  # local pool indices
        selected_hashes = set()
        n_total_sampled = 0
        n_out_of_pool = 0

        while len(collected_X) < self.q:
            torch.set_default_dtype(torch.float32)
            trajectories = self.forward_sampler.sample_trajectories_batch(
                n_total_trajectories=self.q, batch_size=self.q
            )
            terminal_states = trajectories.get_last_states_flat()
            torch.set_default_dtype(prev_dtype)

            for state in terminal_states:
                if isinstance(state, ReactionStateEarlyTerminal):
                    continue
                n_total_sampled += 1
                fp = smiles_to_morgan_fp(state.molecule.smiles)
                key = fp.numpy().tobytes()

                if key not in pool_map:
                    n_out_of_pool += 1
                    continue
                if key in selected_hashes:
                    continue

                selected_hashes.add(key)
                collected_X.append(fp)
                collected_idx.append(pool_map[key])

                if len(collected_X) == self.q:
                    break

            remaining = len(pool_map) - len(selected_hashes)
            if remaining == 0:
                print(
                    f"Warning: pool exhausted after collecting {len(collected_X)}/{self.q} candidates"
                )
                break

        frac_in_pool = 1 - (n_out_of_pool / max(n_total_sampled, 1))
        if frac_in_pool < 1.0:
            print(f"Warning: {n_out_of_pool}/{n_total_sampled} GFN samples were out-of-pool")

        X = torch.stack(collected_X).to(candidates.device)  # (M, 2048)

        # Evaluate acq over M candidates and select top-q
        with torch.no_grad():
            acq_values = acq_func(X.reshape(-1, 1, X.shape[-1]))  # (q,)

        # JS divergence: GFN empirical vs true acq distribution over full unobserved pool
        if self.compute_js_divergence:
            with torch.no_grad():
                true_acq = torch.cat(
                    [
                        acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                        for chunk in candidates.split(self.max_batch_size)
                    ]
                )
            true_acq = true_acq.clamp(min=0)
            true_dist = true_acq / true_acq.sum()

            eval_counts = torch.zeros(len(candidates), device=candidates.device)

            torch.set_default_dtype(torch.float32)
            trajectories = self.forward_sampler.sample_trajectories_batch(
                n_total_trajectories=self.M_eval, batch_size=self.M_eval
            )
            terminal_states = trajectories.get_last_states_flat()
            torch.set_default_dtype(prev_dtype)

            for state in terminal_states:
                if isinstance(state, ReactionStateEarlyTerminal):
                    continue
                fp = smiles_to_morgan_fp(state.molecule.smiles)
                key = fp.numpy().tobytes()
                if key in pool_map:
                    eval_counts[pool_map[key]] += 1

            gfn_dist = eval_counts / eval_counts.sum().clamp(min=1)
            js = _compute_js_divergence(true_dist, gfn_dist)
        else:
            js = None

        return OptimizationResult(
            new_X=X,
            acq_val=acq_values,
            smiles=None,
            metrics={
                "js_divergence": js,
                "frac_in_pool": frac_in_pool,
            },
        )

    def sample_init(self, oracle, n_init: int) -> Initialization:
        uniform_sampler = RandomSampler(
            env=self.forward_sampler.env,
            policy=UniformPolicy(),
            reward=None,
        )

        sampled_states = []
        sampled_smiles = set()
        while len(sampled_states) < n_init:
            trajectories = uniform_sampler.sample_trajectories_batch(
                n_total_trajectories=n_init - len(sampled_states),
                batch_size=n_init - len(sampled_states),
            )
            for state in trajectories.get_last_states_flat():
                if (
                    not isinstance(state, ReactionStateEarlyTerminal)
                    and state.molecule.smiles not in sampled_smiles
                ):
                    sampled_states.append(state)
                    sampled_smiles.add(state.molecule.smiles)

        smiles = [s.molecule.smiles for s in sampled_states]
        train_X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])
        train_y = oracle(train_X)
        observed_indices = [
            oracle._hash_to_idx[hash(row.cpu().numpy().tobytes())] for row in train_X
        ]

        return Initialization(
            train_X=train_X,
            train_y=train_y,
            smiles=smiles,
            observed_indices=observed_indices,
        )
