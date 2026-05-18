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

from molbo.model.base import SurrogateModel


class Acquisition(ABC):
    """Wrapper for BoTorch acquisition functions."""

    def update(self, model: SurrogateModel):
        """Update the acquisition function with a fitted surrogate model."""
        if not model.is_fitted:
            raise RuntimeError("Acquisition received an unfitted model. Call model.fit() first.")
        self.model = model
        self._update()

    @abstractmethod
    def _update(self):
        """Build the BoTorch acquisition function from self.model."""
        ...

    def __call__(self, X):
        X_f = self.model.featurizer(X)
        return self.acq_func(X_f.reshape(-1, 1, X_f.shape[-1]))


class EIAcquisition(Acquisition):
    """Expected improvement acquisition function."""

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = ExpectedImprovement(model=self.model.model, best_f=self.best_f)


class LogEIAcquisition(Acquisition):
    """Log expected improvement acquisition function."""

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = LogExpectedImprovement(model=self.model.model, best_f=self.best_f)


class PIAcquisition(Acquisition):
    """Probability of improvement acquisition function."""

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = ProbabilityOfImprovement(model=self.model.model, best_f=self.best_f)


class LogPIAcquisition(Acquisition):
    """Log probability of improvement acquisition function."""

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = LogProbabilityOfImprovement(model=self.model.model, best_f=self.best_f)


class UCBAcquisition(Acquisition):
    """UCB acquisition function."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def _update(self):
        self.acq_func = UpperConfidenceBound(model=self.model.model, beta=self.beta)


class PosteriorMeanAcquisition(Acquisition):
    """Posterior mean acquisition function."""

    def _update(self):
        self.acq_func = PosteriorMean(model=self.model.model)


class TSAcquisition(Acquisition):
    """Thompson sampling acquisition function."""

    def _update(self):
        self.acq_func = PathwiseThompsonSampling(self.model.model)


class KGAcquisition(Acquisition):
    """Knowledge gradient acquisition function."""

    def __init__(self, num_fantasies: int = 4):
        self.num_fantasies = num_fantasies

    def _update(self):
        with torch.no_grad():
            current_value = self.model.model.posterior(self.model.model.train_inputs[0]).mean.max()

        self.acq_func = qKnowledgeGradient(
            model=self.model.model, num_fantasies=self.num_fantasies, current_value=current_value
        )
