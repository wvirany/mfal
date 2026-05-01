from abc import ABC, abstractmethod

import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from molbo.model.base import SurrogateModel


class Acquisition(ABC):
    """Wrapper for BoTorch acquisition functions."""

    @abstractmethod
    def update(self, model: SurrogateModel):
        """Updaate acquisition function with new surrogate model."""
        pass

    def __call__(self, X):
        return self.acq_func(X)


class EIAcquisition(Acquisition):
    """Expected improvement acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ExpectedImprovement(model=model.model, best_f=self.best_f)


class LogEIAcquisition(Acquisition):
    """Log expected improvement acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogExpectedImprovement(model=model.model, best_f=self.best_f)


class PIAcquisition(Acquisition):
    """Probability of improvement acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class LogPIAcquisition(Acquisition):
    """Log probability of improvement acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class UCBAcquisition(Acquisition):
    """UCB acquisition function."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = UpperConfidenceBound(model=model.model, beta=self.beta)


class PosteriorMeanAcquisition(Acquisition):
    """Posterior mean acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = PosteriorMean(model=model.model)


class TSAcquisition(Acquisition):
    """Thompson sampling acquisition function."""

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = PathwiseThompsonSampling(model.model)


class KGAcquisition(Acquisition):
    """Knowledge gradient acquisition function."""

    def __init__(self, num_fantasies: int = 4):
        self.num_fantasies = num_fantasies

    def update(self, model: SurrogateModel):
        self.model = model

        with torch.no_grad():
            current_value = model.model.posterior(model.model.train_inputs[0]).mean.max()

        self.acq_func = qKnowledgeGradient(
            model=model.model, num_fantasies=self.num_fantasies, current_value=current_value
        )


class ProbabilityOfOptimality(Acquisition):

    def __init__(
        self,
        n_representer_points: int = 100,
        n_samples: int = 500,
    ):
        self.n_representer_points = n_representer_points
        self.n_samples = n_samples
        self.norm = torch.distributions.Normal(0, 1)

    def update(self, model: SurrogateModel, candidates: torch.Tensor = None):
        self.model = model

        if candidates is not None:
            self.candidates = candidates

        # Sample representer points
        if self.candidates is None:
            raise ValueError("Candidates need to be passed before calling update()")
        idx = torch.randperm(len(self.candidates))[: self.n_representer_points]
        X_rep = self.candidates[idx]

        with torch.no_grad():
            mean_rep, std_rep = model(X_rep)
        mean_rep = mean_rep.squeeze()
        std_rep = std_rep.squeeze()

        self.taus = self._sample_taus(mean_rep, std_rep)

    def _approx_cdf(self, z, mean, std):
        return self.norm.cdf((z - mean) / std).prod()

    def _find_percentile(self, u, mean, std):
        lo = (mean - 3 * std).min().item()
        hi = (mean + 3 * std).max().item()
        for _ in range(50):
            mid = (lo + hi) / 2
            if self._approx_cdf(mid, mean, std) < u:
                lo = mid
            else:
                hi = mid
        return mid

    def _sample_taus(self, mean, std):
        y1 = self._find_percentile(0.25, mean, std)
        y2 = self._find_percentile(0.75, mean, std)

        c1 = -torch.log(-torch.log(torch.tensor(0.25)))
        c2 = -torch.log(-torch.log(torch.tensor(0.75)))

        b = (y2 - y1) / (c2 - c1)
        a = y1 - b * c1

        u_samples = torch.rand(self.n_samples)
        taus = a - b * torch.log(-torch.log(u_samples))
        return taus.to(mean.device)

    def __call__(self, X):
        # X: (batch, 1, d) from PoolSampler
        X_squeezed = X.reshape(-1, X.shape[-1])  # (batch, d)

        with torch.no_grad():
            mean, std = self.model(X_squeezed)
        mean = mean.squeeze(-1)  # (batch,)
        std = std.squeeze(-1)  # (batch,)

        r_x = self.norm.cdf((mean.unsqueeze(0) - self.taus.unsqueeze(-1)) / std.unsqueeze(0)).mean(
            dim=0
        )  # (batch,)

        return r_x
