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
        return new_X, acq_val, None


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

        n_nonzero = (probs > 0).sum().item()
        if n_nonzero < self.q:
            print(
                f"Warning: only {n_nonzero} non-zero probability candidates, falling back to replacement sampling"
            )
            idx = torch.multinomial(probs, num_samples=self.q, replacement=True)
        else:
            idx = torch.multinomial(probs, num_samples=self.q, replacement=False)

        new_X = candidates[idx]
        acq_val = acq_values[idx]
        return new_X, acq_val, acq_values


class ThompsonSampler(AcqfOptimizer):
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        self.q = q
        self.max_batch_size = max_batch_size

    def optimize(self, acq_func, candidates: torch.Tensor):
        selected_X = []
        selected_vals = []
        remaining = candidates

        for _ in range(self.q):
            means, stds = [], []
            for chunk in remaining.split(self.max_batch_size):
                with torch.no_grad():
                    posterior = acq_func.model.model.posterior(chunk)
                    means.append(posterior.mean.squeeze(-1))
                    stds.append(posterior.variance.squeeze(-1).sqrt())

            mean = torch.cat(means)
            std = torch.cat(stds)
            sample = mean + std * torch.randn_like(mean)

            idx = sample.argmax()
            selected_X.append(remaining[idx])
            selected_vals.append(sample[idx])

            mask = torch.ones(len(remaining), dtype=torch.bool)
            mask[idx] = False
            remaining = remaining[mask]

        return torch.stack(selected_X), torch.stack(selected_vals), None
