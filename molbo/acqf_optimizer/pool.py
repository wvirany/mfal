import torch
from botorch.optim import optimize_acqf_discrete

from molbo.acqf_optimizer.base import AcqfOptimizer


class PoolMaximizer(AcqfOptimizer):
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        self.q = q
        self.max_batch_size = max_batch_size

    def optimize(self, acq_func, candidates: torch.Tensor):
        new_X, acq_val = optimize_acqf_discrete(
            acq_function=acq_func.acq_func,
            q=self.q,
            choices=candidates,
            max_batch_size=self.max_batch_size,
        )
        return new_X, acq_val


class PoolSampler(AcqfOptimizer):
    def __init__(self, max_batch_size: int = 1024):
        self.max_batch_size = max_batch_size

    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in candidates.split(self.max_batch_size)
                ]
            )
        probs = acq_values / acq_values.sum()
        idx = torch.multinomial(probs, num_samples=1)
        new_X = candidates[idx]
        acq_val = acq_values[idx]
        return new_X, acq_val
