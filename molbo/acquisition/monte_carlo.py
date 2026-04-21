import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

from molbo.acquisition import Acquisition
from molbo.model import SurrogateModel


class qEIAcquisition(Acquisition):
    """Monte Carlo expected improvement acquisition function."""

    def __init__(self, num_samples: int = 512):
        self.num_samples = num_samples

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.acq_func = qExpectedImprovement(model=model.model, best_f=self.best_f, sampler=sampler)


class qUCBAcquisition(Acquisition):
    """Monte Carlo UCB acquisition function."""

    def __init__(self, beta: float = 1.0, num_samples: int = 512):
        self.beta = beta
        self.num_samples = num_samples

    def update(self, model: SurrogateModel):
        self.model = model
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.acq_func = qUpperConfidenceBound(model=model.model, beta=self.beta, sampler=sampler)
