# Suppress warnings - these are safe to ignore for molecular fingerprints
import warnings

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from molbo.models.base import SurrogateModel

warnings.filterwarnings("ignore")


class GPModel(SurrogateModel):
    """A wrapper for SingleTaskGP that implements the SurrogateModel interface"""

    def __init__(self, state_dict=None):
        self.train_X = torch.tensor([])
        self.train_y = torch.tensor([])

    def initialize(self, train_X, train_y, state_dict=None):

        self.train_X = train_X
        self.train_y = train_y

        self.model = SingleTaskGP(train_X, train_y, input_transform=Normalize(d=train_X.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def fit(self):
        fit_gpytorch_mll(self.mll)

    def update(self, new_X, new_y):
        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self.initialize(self.train_X, self.train_y)

    def __call__(self, X):
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            return posterior.mean, posterior.stddev

    def loss(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.train_X)
            return self.mll(output, self.train_y.squeeze())


class TanimotoGP(SingleTaskGP):
    """GP with min-max (Tanimoto) kernel for molecular fingerprints."""

    def __init__(self, train_X, train_y):
        super().__init__(train_X, train_y)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())


class TanimotoGPModel(GPModel):
    """Wrapper for TanimotoGP model."""

    def initialize(self, train_X, train_y, state_dict=None):

        self.train_X = train_X
        self.train_y = train_y

        self.model = TanimotoGP(train_X, train_y)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
