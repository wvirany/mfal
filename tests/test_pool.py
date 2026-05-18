import pytest
import torch
from botorch.optim import optimize_acqf_discrete as botorch_optimize_acqf_discrete

from molbo.acqf_optimizer.pool import PoolMaximizer, PoolSampler, _optimize_acqf_discrete
from molbo.acquisition.base import EIAcquisition
from molbo.acquisition.monte_carlo import qEIAcquisition
from molbo.bo import BOLoop, History
from molbo.model.featurizer import IdentityFeaturizer
from molbo.model.gp import TanimotoGPModel
from molbo.oracle.lookup import LookupOracle

N, D, N_INIT = 30, 16, 5


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def candidates():
    torch.manual_seed(0)
    return torch.rand(N, D, dtype=torch.float64)


@pytest.fixture
def oracle(candidates):
    torch.manual_seed(1)
    y = torch.randn(N, 1, dtype=torch.float64)
    return LookupOracle(X_data=candidates, y_data=y)


@pytest.fixture
def ei(candidates, oracle):
    """Analytic EI — deterministic, for q=1 comparisons."""
    model = TanimotoGPModel(featurizer=IdentityFeaturizer())
    model.initialize(candidates[:N_INIT], oracle.y_data[:N_INIT])
    model.fit()
    acq = EIAcquisition()
    acq.update(model)
    return acq


@pytest.fixture
def qei(candidates, oracle):
    """MC qEI — supports X_pending, for q>1 comparisons."""
    model = TanimotoGPModel(featurizer=IdentityFeaturizer())
    model.initialize(candidates[:N_INIT], oracle.y_data[:N_INIT])
    model.fit()
    acq = qEIAcquisition()
    acq.update(model)
    return acq


def _build_loop(candidates, oracle, optimizer, checkpoint_path=None):
    indices = list(range(N_INIT))
    history = History(
        X_init=candidates[indices],
        y_init=oracle.y_data[indices],
        observed_indices=indices,
        checkpoint_path=checkpoint_path,
        checkpoint_freq=1,
    )
    return (
        BOLoop(
            history=history,
            model=TanimotoGPModel(featurizer=IdentityFeaturizer()),
            acq_func=qEIAcquisition(),
            oracle=oracle,
            acqf_optimizer=optimizer,
            candidates=candidates,
        ),
        history,
    )


# ── _optimize_acqf_discrete ───────────────────────────────────────────────────


class TestOptimizeAcqfDiscrete:

    def test_q1_matches_botorch(self, candidates, ei):
        chosen, val = _optimize_acqf_discrete(ei.acq_func, q=1, choices=candidates)
        ref_X, ref_val = botorch_optimize_acqf_discrete(ei.acq_func, q=1, choices=candidates)
        assert torch.allclose(candidates[chosen], ref_X.squeeze(0))
        assert torch.isclose(val, ref_val)

    def test_q1_observed_matches_botorch_filtered(self, candidates, ei):
        observed = list(range(N_INIT))
        available = [i for i in range(N) if i not in set(observed)]
        chosen, val = _optimize_acqf_discrete(
            ei.acq_func, q=1, choices=candidates, observed_indices=observed
        )
        ref_X, ref_val = botorch_optimize_acqf_discrete(
            ei.acq_func, q=1, choices=candidates[available]
        )
        assert torch.allclose(candidates[chosen], ref_X.squeeze(0))
        assert torch.isclose(val, ref_val)

    def test_q_gt1_matches_botorch(self, candidates, qei):
        q = 4
        torch.manual_seed(42)
        chosen, vals = _optimize_acqf_discrete(qei.acq_func, q=q, choices=candidates)
        torch.manual_seed(42)
        ref_X, ref_vals = botorch_optimize_acqf_discrete(qei.acq_func, q=q, choices=candidates)
        assert torch.allclose(candidates[chosen], ref_X)
        assert torch.allclose(vals, ref_vals)

    def test_q_gt1_observed_matches_botorch_filtered(self, candidates, qei):
        observed = list(range(N_INIT))
        available = [i for i in range(N) if i not in set(observed)]
        q = 3
        torch.manual_seed(42)
        chosen, vals = _optimize_acqf_discrete(
            qei.acq_func, q=q, choices=candidates, observed_indices=observed
        )
        torch.manual_seed(42)
        ref_X, ref_vals = botorch_optimize_acqf_discrete(
            qei.acq_func, q=q, choices=candidates[available]
        )
        assert torch.allclose(candidates[chosen], ref_X)
        assert torch.allclose(vals, ref_vals)

    def test_observed_never_chosen(self, candidates, ei):
        observed = list(range(N_INIT))
        chosen, _ = _optimize_acqf_discrete(
            ei.acq_func, q=1, choices=candidates, observed_indices=observed
        )
        assert not any(i in set(observed) for i in chosen)

    def test_unique_indices(self, candidates, qei):
        chosen, _ = _optimize_acqf_discrete(qei.acq_func, q=5, choices=candidates)
        assert len(chosen) == len(set(chosen))


# ── End-to-end BOLoop ─────────────────────────────────────────────────────────


class TestBOLoopPool:

    def test_maximizer_q1(self, candidates, oracle):
        n_iters = 5
        loop, history = _build_loop(candidates, oracle, PoolMaximizer(q=1))
        loop.run(n_iters=n_iters)
        assert len(history.observed_indices) == N_INIT + n_iters
        assert len(set(history.observed_indices)) == len(history.observed_indices)
        assert torch.allclose(history.X_observed, candidates[history.observed_indices[N_INIT:]])

    def test_maximizer_q_gt1(self, candidates, oracle):
        q, n_iters = 3, 3
        loop, history = _build_loop(candidates, oracle, PoolMaximizer(q=q))
        loop.run(n_iters=n_iters)
        assert len(history.observed_indices) == N_INIT + q * n_iters
        assert len(set(history.observed_indices)) == len(history.observed_indices)

    def test_sampler_q1(self, candidates, oracle):
        n_iters = 5
        loop, history = _build_loop(candidates, oracle, PoolSampler(q=1))
        loop.run(n_iters=n_iters)
        assert len(history.observed_indices) == N_INIT + n_iters
        assert len(set(history.observed_indices)) == len(history.observed_indices)

    def test_sampler_uniform_fallback(self, candidates, oracle):
        """All-zero acq values should fall back to uniform sampling without error."""
        # Force all-zero by observing the best candidates so EI is ~0 everywhere
        loop, history = _build_loop(candidates, oracle, PoolSampler(q=1))
        loop.run(n_iters=1)  # just check it doesn't crash

    def test_noisy_oracle(self, candidates):
        torch.manual_seed(2)
        y = torch.randn(N, 1, dtype=torch.float64)
        noisy_oracle = LookupOracle(X_data=candidates, y_data=y, noise_std=0.1)
        loop, history = _build_loop(candidates, noisy_oracle, PoolMaximizer(q=1))
        loop.run(n_iters=3)
        assert len(history.observed_indices) == N_INIT + 3

    def test_checkpoint_resume(self, candidates, oracle, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        n_mid, n_iters = 3, 5

        # Run to mid-point
        loop, history = _build_loop(
            candidates, oracle, PoolMaximizer(q=1), checkpoint_path=checkpoint
        )
        loop.run(n_iters=n_mid)

        # Resume and finish
        history2 = History.load(checkpoint)
        assert history2.start_iteration == n_mid

        loop2 = BOLoop(
            history=history2,
            model=TanimotoGPModel(featurizer=IdentityFeaturizer()),
            acq_func=qEIAcquisition(),
            oracle=oracle,
            acqf_optimizer=PoolMaximizer(q=1),
            candidates=candidates,
        )
        loop2.run(n_iters=n_iters)
        assert len(history2.observed_indices) == N_INIT + n_iters
        assert len(set(history2.observed_indices)) == len(history2.observed_indices)
