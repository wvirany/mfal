import torch
from botorch.optim import optimize_acqf

from molbo.acqf_optimizer.base import AcqfOptimizer, Initialization, OptimizationResult


class ContinuousBase(AcqfOptimizer):

    def sample_init(self, oracle, n_init: int) -> Initialization:
        self.bounds = oracle.bounds
        train_X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
            n_init, self.bounds.shape[1], dtype=torch.float64, device=self.bounds.device
        )
        train_y = oracle(train_X)
        return Initialization(train_X=train_X, train_y=train_y)


class ContinuousMaximizer(ContinuousBase):
    def __init__(self, q: int = 1, num_restarts: int = 5, raw_samples: int = 20):
        self.q = q
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def _optimize(self, acq_func, candidates=None, observed_indices=None):
        new_X, acq_val = optimize_acqf(
            acq_function=acq_func.acq_func,
            bounds=self.bounds,
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return OptimizationResult(new_X=new_X, acq_val=acq_val)


class ContinuousSampler(ContinuousBase):
    def __init__(self, q: int = 1, n_samples: int = 1000):
        self.q = q
        self.n_samples = n_samples

    def _optimize(self, acq_func, candidates=None, observed_indices=None):
        X_grid = torch.linspace(
            self.bounds[0].item(),
            self.bounds[1].item(),
            self.n_samples,
            dtype=torch.float64,
        ).unsqueeze(-1)
        with torch.no_grad():
            acq_values = acq_func(X_grid.reshape(-1, 1, 1))

        # Guard against the case when acquisition values are negative
        if (acq_values < 0).any():
            print("Warning: negative acquisition values encountered during sampling")
            print(f"Total acq_vals < 0: {(acq_values < 0).sum()}")
        assert not (acq_values < 0).all(), "All acquisition values are negative"
        assert acq_values.sum() > 0, "Normalization constant is negative; exiting"

        # Clamp negative values
        acq_clamped = acq_values.clamp(min=0)
        probs = acq_clamped / acq_clamped.sum()

        idx = torch.multinomial(probs, num_samples=self.q, replacement=False)
        return OptimizationResult(new_X=X_grid[idx], acq_val=acq_values[idx])
