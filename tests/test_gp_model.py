"""Test GP model initialization and basic functionality."""

import torch
from botorch import fit_gpytorch_mll

from mfal.models.gp import TanimotoGP, initialize_gp

# Generate random data for testing
probs = torch.ones(10, 2048, dtype=torch.float64) / 2
train_x = torch.bernoulli(probs)
train_y = torch.randn(10, 1, dtype=torch.float64)

test_x = torch.bernoulli(probs)
test_y = torch.randn(10, 1, dtype=torch.float64)


def test_model_creation():
    """Test GP model creation."""

    model = TanimotoGP(train_x, train_y)
    assert model is not None


def test_gp_initialization():
    """Test GP model initialization."""

    model, mll = initialize_gp(train_x, train_y)

    assert model is not None
    assert mll is not None


def test_gp_training():
    """Test GP model training."""

    model, mll = initialize_gp(train_x, train_y)

    # Fit hyperparameters
    fit_gpytorch_mll(mll)


def test_gp_prediction():
    """Test GP model prediction."""

    model, mll = initialize_gp(train_x, train_y)

    # Fit hyperparameters
    fit_gpytorch_mll(mll)

    # Make predictions
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean
        var = posterior.variance
        covar = posterior.covariance_matrix

    assert mean.shape == (10, 1)
    assert var.shape == (10, 1)
    assert torch.all(var > 0)
    assert covar.shape == (10, 10)
    assert torch.all(covar > 0)
