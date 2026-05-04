import torch
from botorch.optim import optimize_acqf_discrete

from molbo.acqf_optimizer.base import AcqfOptimizer, Initialization, OptimizationResult
from molbo.acquisition import Acquisition
from molbo.utils import get_centroid_indices_from_fps


def compute_acq_sparsity(acq_values: torch.Tensor) -> float:
    return 1 - (acq_values.mean() / acq_values.max()).item()


def compute_acq_entropy(acq_values: torch.Tensor) -> float:
    shifted = acq_values - acq_values.min()
    probs = shifted / shifted.sum()
    return -(probs * torch.log(probs + 1e-10)).sum().item()


class PoolBase(AcqfOptimizer):
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        self.q = q
        self.max_batch_size = max_batch_size

    def sample_init(self, oracle, n_init: int) -> Initialization:
        indices = torch.randperm(len(oracle.candidates))[:n_init].tolist()
        train_X, train_y = oracle[indices]
        return Initialization(train_X=train_X, train_y=train_y, observed_indices=indices)


class PoolMaximizer(PoolBase):
    def optimize(self, acq_func, candidates: torch.Tensor):
        new_X, acq_val = optimize_acqf_discrete(
            acq_function=acq_func.acq_func,
            q=self.q,
            choices=candidates,
            max_batch_size=self.max_batch_size,
        )
        return OptimizationResult(new_X=new_X, acq_val=acq_val)


class PoolSampler(PoolBase):
    def optimize(self, acq_func, candidates: torch.Tensor):
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
            metrics={
                "acq_sparsity": compute_acq_sparsity(acq_values),
                "acq_entropy": compute_acq_entropy(acq_values),
            },
        )


class ThompsonSampler(PoolBase):
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

        return OptimizationResult(
            new_X=torch.stack(selected_X),
            acq_val=torch.stack(selected_vals),
        )


class TopKSelector(PoolBase):
    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in candidates.split(self.max_batch_size)
                ]
            )
        top_q = acq_values.topk(min(self.q, len(candidates)))
        return OptimizationResult(new_X=candidates[top_q.indices], acq_val=top_q.values)


class TopKModesSelector(PoolBase):
    """Select top-q candidates by greedy Tanimoto clustering on acquisition values."""

    def __init__(self, q: int = 50, max_batch_size: int = 1024, tanimoto_threshold: float = 0.7):
        super().__init__(q=q, max_batch_size=max_batch_size)
        self.tanimoto_threshold = tanimoto_threshold

    def optimize(self, acq_func, candidates: torch.Tensor):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in candidates.split(self.max_batch_size)
                ]
            )
        centroid_indices = get_centroid_indices_from_fps(
            candidates, acq_values, self.tanimoto_threshold
        )
        # take top-q centroids (already in descending score order)
        selected = centroid_indices[: self.q]
        idx = torch.tensor(selected, device=candidates.device)
        return OptimizationResult(new_X=candidates[idx], acq_val=acq_values[idx])


class GreedySampler(AcqfOptimizer):
    """Sample M candidates using any optimizer, then select q greedily using a downstream optimizer."""

    def __init__(
        self,
        sampler: AcqfOptimizer,
        selector: AcqfOptimizer,
        sampler_acquisition: Acquisition,
        selector_acquisition: Acquisition,
    ):
        self.sampler = sampler
        self.selector = selector
        self.sampler_acquisition = sampler_acquisition
        self.selector_acquisition = selector_acquisition

    def sample_init(self, oracle, n_init):
        return self.sampler.sample_init(oracle, n_init)

    def optimize(self, acq_func, candidates: torch.Tensor):
        model = acq_func.model
        self.sampler_acquisition.update(model)
        self.selector_acquisition.update(model)
        sample_result = self.sampler.optimize(self.sampler_acquisition, candidates)
        select_result = self.selector.optimize(self.selector_acquisition, sample_result.new_X)
        return select_result
