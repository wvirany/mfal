# Suppress warnings - these are safe to ignore for molecular fingerprints
import warnings

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from molbo.models import SurrogateModel
from molbo.models.modules import FixedNoise, FixedObservationMean, FixedRBFKernel

warnings.filterwarnings("ignore")


class GPModel(SurrogateModel):
    """A wrapper for SingleTaskGP that implements the SurrogateModel interface"""

    def __init__(self, mean_module=None, covar_module=None, noise_module=None):
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.noise_module = noise_module

    def initialize(self, train_X, train_y, state_dict=None):

        self.train_X = train_X
        self.train_y = train_y

        self.model = SingleTaskGP(
            train_X,
            train_y,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            input_transform=Normalize(d=train_X.shape[-1]),
        )

        self._init_modules()

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def _init_modules(self):
        # Note this also updates transform parameters
        X_transformed, y_transformed = self.train_X, self.train_y
        if self._input_transform is not None:
            X_transformed = self.model.input_transform(self.train_X)
        if self._outcome_transform is not None:
            y_transformed, _ = self.model.outcome_transform(self.train_y)

        if isinstance(self.mean_module, FixedObservationMean):
            self.mean_module.initialize_params(y_transformed)

        if isinstance(self.covar_module, FixedRBFKernel):
            self.covar_module.initialize_params(X_transformed)

        if isinstance(self.noise_module, FixedNoise):
            self.noise_module.initialize_params(y_transformed, self.model)

    def fit(self):
        # Only fit if there are trainable parameters
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
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

    @property
    def _input_transform(self):
        """Return SingleTaskGP input_transform if it exists, else None"""
        return getattr(self.model, "input_transform", None)

    @property
    def _outcome_transform(self):
        """Return SingleTaskGP outcome_transform if it exists, else None"""
        return getattr(self.model, "outcome_transform", None)


class TanimotoGP(SingleTaskGP):

    def __init__(self, train_X, train_y, mean_module=None):
        super().__init__(
            train_X,
            train_y,
            mean_module=mean_module,
            covar_module=gpytorch.kernels.ScaleKernel(TanimotoKernel()),
        )


class TanimotoGPModel(GPModel):
    """Wrapper for TanimotoGP model."""

    def initialize(self, train_X, train_y, state_dict=None):

        self.train_X = train_X
        self.train_y = train_y

        self.model = TanimotoGP(
            train_X,
            train_y,
            mean_module=self.mean_module,
        )

        self._init_modules()

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
