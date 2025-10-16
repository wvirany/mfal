from botorch.models.gp_regression import SingleTaskGP
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood


class TanimotoGP(SingleTaskGP):
    """GP model with MinMax kernel for Morgan fingerprints."""

    def __init__(self, train_x, train_y):
        super().__init__(train_x, train_y, likelihood=GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_x)  # Ensure model parameters are on the same device as the data

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def initialize_gp(train_x, train_y, state_dict=None):
    """
    Initialize GP model and loss function.

    Args:
        train_x: Tensor of shape (n_samples, n_features)
        train_y: Tensor of shape (n_samples,)
        state_dict: Optional state dictionary to load model checkpoint

    Returns: mll object, model object
    """

    model = TanimotoGP(train_x, train_y).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Load state dictionary if provided
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, mll
