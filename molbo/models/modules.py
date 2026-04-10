"""
Modules for specifying fixed GP parameters.

These don't get updated when fit() is called by BOLoop, but are updated by GPModel._init_modules()
"""

import gpytorch
import torch

from molbo.oracle import mcl1_vina

torch.set_default_dtype(torch.float64)


class FixedObservationMean(gpytorch.means.ConstantMean):
    def __init__(self):
        super().__init__()

    def initialize_params(self, train_y):
        self.constant = train_y.mean()
        self.constant.requires_grad_(False)


class FixedObservationMax(FixedObservationMean):
    def initialize_params(self, train_y):
        self.constant = train_y.max()
        self.constant.requires_grad_(False)


class FixedObservationMin(FixedObservationMean):
    def initialize_params(self, train_y):
        self.constant = train_y.min()
        self.constant.requires_grad_(False)


class Mcl1DockingMean(gpytorch.means.Mean):
    """Docking scores for Mcl1 dataset used as prior mean for MM/GBSA"""

    def __init__(self):
        super().__init__()
        self.oracle = mcl1_vina()

    def forward(self, x):
        x_flat = x.reshape(-1, x.shape[-1]).cpu()
        # Reshape to leading dimensions (i.e., drop 'd' dimension)
        res = self.oracle(x_flat).to(x.device).reshape(x.shape[:-1])
        return res


class FixedRBFKernel(gpytorch.kernels.RBFKernel):
    def __init__(self, lengthscale=None):
        super().__init__()
        self._lengthscale = lengthscale

    def initialize_params(self, train_X):
        if self._lengthscale is not None:
            value = self._lengthscale
        else:
            median_dist = torch.cdist(train_X, train_X).median()
            value = median_dist.clamp(min=1e-3)
        self.lengthscale = value
        self.raw_lengthscale.requires_grad_(False)


class FixedNoise:
    def __init__(self, noise=None):
        self._noise = noise

    def initialize_params(self, train_y, model):
        if self._noise is not None:
            value = self._noise
        else:
            value = (train_y.var() * 0.1).clamp(min=1e-4)

        model.likelihood.noise_covar.noise = value
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)
