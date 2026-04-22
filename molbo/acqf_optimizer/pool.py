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
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        self.q = q
        self.max_batch_size = max_batch_size

    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in candidates.split(self.max_batch_size)
                ]
            )

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
        new_X = candidates[idx]
        acq_val = acq_values[idx]
        return new_X, acq_val
