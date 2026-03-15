"""
Modules for specifying fixed GP parameters.

These don't get updated when fit() is called by BOLoop.
"""

import gpytorch
import torch

torch.set_default_dtype(torch.float64)


class FixedMean(gpytorch.means.ConstantMean):

    def __init__(self, constant=None):
        super().__init__()
        self._constant = constant

    def initialize_params(self, train_y):
        value = torch.tensor(self._constant) if self._constant is not None else train_y.mean()
        self.constant = value
        self.constant.requires_grad_(False)


class FixedObservationMax(FixedMean):
    def initialize_params(self, train_y):
        self.constant = train_y.max()
        self.constant.requires_grad_(False)


class FixedObservationMin(FixedMean):
    def initialize_params(self, train_y):
        self.constant = train_y.min()
        self.constant.requires_grad_(False)


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
