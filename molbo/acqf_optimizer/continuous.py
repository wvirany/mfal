import torch
from botorch.optim import optimize_acqf

from molbo.acqf_optimizer.base import AcqfOptimizer


class ContinuousMaximizer(AcqfOptimizer):
    def __init__(self, num_restarts: int = 5, raw_samples: int = 20):
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def optimize(self, acq_func, candidates=None):
        assert self.bounds is not None, "bounds must be set before calling optimize"
        new_X, acq_val = optimize_acqf(
            acq_function=acq_func.acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return new_X, acq_val, None


class ContinuousSampler(AcqfOptimizer):
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples

    def optimize(self, acq_func, candidates=None):
        X_grid = torch.linspace(
            self.bounds[0].item(),
            self.bounds[1].item(),
            self.n_samples,
            dtype=torch.float64,
        ).unsqueeze(-1)
        with torch.no_grad():
            acq_values = acq_func(X_grid.reshape(-1, 1, 1)).squeeze()
        probs = acq_values / acq_values.sum()
        idx = torch.multinomial(probs, num_samples=1).item()
        return X_grid[idx].unsqueeze(0), acq_values[idx], None
