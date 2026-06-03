import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import (
    IIDNormalSampler as BotorchIIDNormalSampler,
)
from botorch.sampling.normal import (
    SobolQMCNormalSampler as BotorchSobolQMCNormalSampler,
)

from molbo.acquisition import Acquisition


class qEIAcquisition(Acquisition):
    """Monte Carlo Expected Improvement acquisition function."""

    def __init__(self, sampler=None):
        self.sampler = sampler

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = qExpectedImprovement(
            model=self.model.model, best_f=self.best_f, sampler=self.sampler
        )


class qLogEIAcquisition(Acquisition):
    """Monte Carlo Log Expected Improvement acquisition function."""

    def __init__(self, sampler=None):
        self.sampler = sampler

    def _update(self):
        self.best_f = self.model.train_y.max().item()
        self.acq_func = qLogExpectedImprovement(
            model=self.model.model, best_f=self.best_f, sampler=self.sampler
        )


class qUCBAcquisition(Acquisition):
    """Monte Carlo Upper Confidence Bound acquisition function."""

    def __init__(self, beta: float = 1.0, sampler=None):
        self.beta = beta
        self.sampler = sampler

    def _update(self):
        self.acq_func = qUpperConfidenceBound(
            model=self.model.model, beta=self.beta, sampler=self.sampler
        )


class IIDNormalSampler(BotorchIIDNormalSampler):
    def __init__(self, num_samples: int = 512, seed=None):
        super().__init__(sample_shape=torch.Size([num_samples]), seed=seed)


class SobolQMCNormalSampler(BotorchSobolQMCNormalSampler):
    def __init__(self, num_samples: int = 512, seed=None):
        super().__init__(sample_shape=torch.Size([num_samples]), seed=seed)


class KBSampler(MCSampler):
    """Kriging Believer sampler - returns posterior mean as a single deterministic fantasy."""

    def __init__(self):
        super().__init__(sample_shape=torch.Size([1]))

    def forward(self, posterior):
        return posterior.mean.unsqueeze(0)
