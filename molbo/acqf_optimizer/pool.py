import torch
from botorch.optim import optimize_acqf_discrete

from molbo.acqf_optimizer.base import AcqfOptimizer, Initialization, OptimizationResult


class PoolBase(AcqfOptimizer):
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        self.q = q
        self.max_batch_size = max_batch_size

    def sample_init(self, oracle, n_init: int) -> Initialization:
        indices = torch.randperm(len(oracle.candidates))[:n_init].tolist()
        train_X, train_y = oracle[indices]
        return Initialization(train_X=train_X, train_y=train_y, observed_indices=indices)


class PoolMaximizer(PoolBase):
    def _optimize(self, acq_func, candidates: torch.Tensor):
        new_X, acq_val = optimize_acqf_discrete(
            acq_function=acq_func.acq_func,
            q=self.q,
            choices=candidates,
            max_batch_size=self.max_batch_size,
        )
        return OptimizationResult(new_X=new_X, acq_val=acq_val)


class PoolSampler(PoolBase):
    def _optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in candidates.split(self.max_batch_size)
                ]
            )

        acq_clamped = acq_values.clamp(min=0)

        if acq_clamped.sum() == 0:
            print(
                "Warning: all acquisition values are zero after clamping; falling back to uniform sampling"
            )
            probs = torch.ones(len(candidates), device=candidates.device) / len(candidates)
        else:
            probs = acq_clamped / acq_clamped.sum()

        n_nonzero = (probs > 0).sum().item()
        if n_nonzero < self.q:
            print(
                f"Warning: only {n_nonzero} non-zero probability candidates, falling back to replacement sampling"
            )
            idx = torch.multinomial(probs, num_samples=self.q, replacement=True)
        else:
            idx = torch.multinomial(probs, num_samples=self.q, replacement=False)

        return OptimizationResult(
            new_X=candidates[idx],
            acq_val=acq_values[idx],
        )
