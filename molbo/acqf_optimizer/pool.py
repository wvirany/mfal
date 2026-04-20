import torch

from molbo.acqf_optimizer.base import AcqfOptimizer


class PoolMaximizer(AcqfOptimizer):
    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = acq_func(candidates.reshape(-1, 1, candidates.shape[-1]))
        local_idx = acq_values.argmax().reshape(1)
        new_X = candidates[local_idx]
        acq_val = acq_values[local_idx]
        return new_X, acq_val, local_idx


class PoolSampler(AcqfOptimizer):
    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = acq_func(candidates.reshape(-1, 1, candidates.shape[-1]))
        probs = acq_values / acq_values.sum()
        local_idx = torch.multinomial(probs, num_samples=1)
        new_X = candidates[local_idx]
        acq_val = acq_values[local_idx]
        return new_X, acq_val, local_idx
