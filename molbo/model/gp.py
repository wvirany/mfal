# Suppress warnings - these are safe to ignore for molecular fingerprints
import warnings

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from molbo.model import SurrogateModel
from molbo.model.featurizer import Featurizer, IdentityFeaturizer
from molbo.model.modules import FixedNoise, FixedObservationMean, FixedRBFKernel

warnings.filterwarnings("ignore")


class GPModel(SurrogateModel):
    """A wrapper for SingleTaskGP that implements the SurrogateModel interface.

    Candidate data is automatically mapped to tensors via a Featurizer object.

    Args:
        mean_module: Optional GP mean module.
        covar_module: Optional GP covariance module.
        noise_module: Optional fixed-noise module.
        featurizer: Maps inputs to feature tensors. Defaults to IdentityFeaturizer
                    (inputs are already feature tensors).
    """

    def __init__(
        self, mean_module=None, covar_module=None, noise_module=None, featurizer: Featurizer = None
    ):
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.noise_module = noise_module
        self.featurizer = featurizer or IdentityFeaturizer()
        self.model = None

    def initialize(self, train_X, train_y, state_dict=None):
        """Featurize and store training data. The GP itself is built in fit()."""

        self.train_X = self.featurizer(train_X)
        self.train_y = train_y

    def _build_model(self):
        """Construct the SingleTaskGP from the current training data."""
        return SingleTaskGP(
            self.train_X,
            self.train_y,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            input_transform=Normalize(d=self.train_X.shape[-1]),
        )

    def fit(self):
        """Build the GP from current data, initialize modules, and fit hyperparameters."""
        self.model = self._build_model()
        self._init_modules()
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Only fit if there are trainable parameters
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
            fit_gpytorch_mll(self.mll)

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

    def update(self, new_X, new_y):
        self.train_X = torch.cat([self.train_X, self.featurizer(new_X)])
        self.train_y = torch.cat([self.train_y, new_y])

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fit. Call fit() before predicting.")

    def __call__(self, X):
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(self.featurizer(X))
            return posterior.mean, posterior.stddev.unsqueeze(-1)

    def loss(self):
        self._check_fitted()
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

    @property
    def is_fitted(self):
        return self.model is not None


class TanimotoGP(SingleTaskGP):

    def __init__(self, train_X, train_y, mean_module=None):
        super().__init__(
            train_X,
            train_y,
            mean_module=mean_module,
            covar_module=gpytorch.kernels.ScaleKernel(TanimotoKernel()),
        )


class TanimotoGPModel(GPModel):
    """GP surrogate model with a Tanimoto kernel.

    Expects pre-featurized inputs (e.g. Morgan fingerprints) by default.
    To featurize SMILES automatically, pass a featurizer explicitly:

    model = TanimotoGPModel(featurizer=MorganFingerprintFeaturizer())
    """

    def _build_model(self):
        return TanimotoGP(self.train_X, self.train_y, mean_module=self.mean_module)
