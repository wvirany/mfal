import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

from molbo.acquisition import Acquisition
from molbo.model.base import SurrogateModel


class qEIAcquisition(Acquisition):
    """Monte Carlo expected improvement acquisition function."""

    def __init__(self, num_samples: int = 512, sampler=None):
        self.num_samples = num_samples
        self.sampler = sampler

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = qExpectedImprovement(
            model=model.model, best_f=self.best_f, sampler=self.sampler
        )


class qUCBAcquisition(Acquisition):
    """Monte Carlo UCB acquisition function."""

    def __init__(self, beta: float = 1.0, num_samples: int = 512, sampler=None):
        self.beta = beta
        self.num_samples = num_samples
        self.sampler = sampler

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = qUpperConfidenceBound(
            model=model.model, beta=self.beta, sampler=self.sampler
        )
