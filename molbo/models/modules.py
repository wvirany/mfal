import gpytorch
import torch

torch.set_default_dtype(torch.float64)


class DataDependentMean(gpytorch.means.ConstantMean):
    def initialize_from_data(self, train_y):
        raise NotImplementedError


class ObservationMean(DataDependentMean):
    def initialize_from_data(self, train_y):
        self.constant = torch.tensor([0.0])
        self.constant.requires_grad_(False)


class ObservationMax(DataDependentMean):
    def initialize_from_data(self, train_y):
        self.constant = (train_y.max() - train_y.mean()) / train_y.std()
        self.constant.requires_grad_(False)


class ObservationMin(DataDependentMean):
    def initialize_from_data(self, train_y):
        self.constant = (train_y.min() - train_y.mean()) / train_y.std()
        self.constant.requires_grad_(False)
