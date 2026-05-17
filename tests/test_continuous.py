"""End-to-end smoke test for the continuous BO setting."""

import torch

from molbo.acqf_optimizer.continuous import ContinuousMaximizer
from molbo.acquisition.base import EIAcquisition
from molbo.bo import BOLoop, History
from molbo.model.gp import GPModel
from molbo.oracle import AnalyticOracle


def _make_oracle():
    def f(X):
        return torch.sin(X).sum(dim=-1, keepdim=True)

    bounds = torch.tensor([[-3.0], [3.0]], dtype=torch.float64)
    return AnalyticOracle(f=f, dim=1, bounds=bounds)


def test_continuous_run_completes():
    oracle = _make_oracle()
    optimizer = ContinuousMaximizer(q=1)
    init = optimizer.sample_init(oracle, n_init=5)

    history = History(X_init=init.train_X, y_init=init.train_y)

    bo_loop = BOLoop(
        history=history,
        model=GPModel(),
        acq_func=EIAcquisition(),
        oracle=oracle,
        acqf_optimizer=optimizer,
    )

    n_iters = 10
    bo_loop.run(n_iters=n_iters)

    # 5 initial points + 10 acquired
    assert history.X_all.shape[0] == 15
    assert history.y_all.shape[0] == 15
    assert len(history.iteration) == n_iters


def test_continuous_resume(tmp_path):
    """A run checkpointed partway should resume and finish at the right place."""
    ckpt = tmp_path / "run.pt"
    oracle = _make_oracle()
    optimizer = ContinuousMaximizer(q=1)
    init = optimizer.sample_init(oracle, n_init=5)

    # First leg: 6 iterations, checkpointing every iteration.
    history = History(
        X_init=init.train_X,
        y_init=init.train_y,
        checkpoint_path=ckpt,
        checkpoint_freq=1,
    )
    BOLoop(
        history=history,
        model=GPModel(),
        acq_func=EIAcquisition(),
        oracle=oracle,
        acqf_optimizer=optimizer,
    ).run(n_iters=6)

    # Resume from checkpoint and run to 10.
    resumed = History.load(ckpt, checkpoint_path=ckpt, checkpoint_freq=1)
    assert resumed is not None
    assert resumed.start_iteration == 6

    BOLoop(
        history=resumed,
        model=GPModel(),
        acq_func=EIAcquisition(),
        oracle=oracle,
        acqf_optimizer=optimizer,
    ).run(n_iters=10)

    assert len(resumed.iteration) == 10
    assert resumed.X_all.shape[0] == 15
