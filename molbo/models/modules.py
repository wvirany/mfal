import gpytorch
import torch

torch.set_default_dtype(torch.float64)


class FixedMean(gpytorch.means.ConstantMean):
    def initialize_from_data(self, train_y):
        raise NotImplementedError


class FixedObservationMean(FixedMean):
    def initialize_from_data(self, train_y):
        self.constant = train_y.mean()
        self.constant.requires_grad_(False)


class FixedObservationMax(FixedMean):
    def initialize_from_data(self, train_y):
        self.constant = train_y.max()
        self.constant.requires_grad_(False)


class FixedObservationMin(FixedMean):
    def initialize_from_data(self, train_y):
        self.constant = train_y.min()
        self.constant.requires_grad_(False)


class FixedRBFKernel(gpytorch.kernels.RBFKernel):
    def initialize_from_data(self, train_X):
        median_dist = torch.cdist(train_X, train_X).median()
        self.lengthscale = median_dist.clamp(min=1e-3)
        self.raw_lengthscale.requires_grad_(False)


class FixedNoise:
    def initialize_from_data(self, train_y, model):
        noise = (train_y.var() * 0.1).clamp(min=1e-4)
        model.likelihood.noise_covar.noise = noise
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)
