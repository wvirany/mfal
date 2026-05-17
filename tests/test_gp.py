"""Lightweight tests for GPModel and TanimotoGPModel.

Checks initialize -> fit -> predict works end-to-end,
the featurizer works (TanimotoGPModel accepts SMILES), and the fit warning works.
"""

import pytest
import torch

from molbo.model.featurizer import IdentityFeaturizer, MorganFingerprintFeaturizer
from molbo.model.gp import GPModel, TanimotoGPModel

# --- Fixtures -------------------------------------------------------------

# Continuous data: feature tensors in, feature tensors out.
CONT_X = torch.rand(8, 3, dtype=torch.float64)
CONT_Y = torch.rand(8, 1, dtype=torch.float64)
CONT_X_NEW = torch.rand(4, 3, dtype=torch.float64)
CONT_Y_NEW = torch.rand(4, 1, dtype=torch.float64)

# Molecular data: SMILES in, the model featurizes internally.
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccncc1"]
SMILES_NEW = ["CCCl", "CCBr"]
MOL_Y = torch.rand(len(SMILES), 1, dtype=torch.float64)
MOL_Y_NEW = torch.rand(len(SMILES_NEW), 1, dtype=torch.float64)


# --- GPModel --------------------------------------------------------------


def test_gpmodel_default_featurizer():
    model = GPModel()
    assert isinstance(model.featurizer, IdentityFeaturizer)


def test_gpmodel_lifecycle():
    model = GPModel()
    model.initialize(CONT_X, CONT_Y)
    model.fit()

    mean, std = model(CONT_X)
    assert mean.shape == (8, 1)
    assert std.shape == (8, 1)
    assert torch.all(std > 0)


def test_gpmodel_update_grows_data():
    model = GPModel()
    model.initialize(CONT_X, CONT_Y)
    model.fit()
    model.update(CONT_X_NEW, CONT_Y_NEW)

    assert model.train_X.shape[0] == 12
    assert model.train_y.shape[0] == 12

    # fit() after update should still work
    model.fit()


def test_gpmodel_loss_runs():
    model = GPModel()
    model.initialize(CONT_X, CONT_Y)
    model.fit()
    loss = model.loss()
    assert torch.is_tensor(loss)


# --- TanimotoGPModel ------------------------------------------------------


def test_tanimoto_default_featurizer():
    model = TanimotoGPModel()
    assert isinstance(model.featurizer, MorganFingerprintFeaturizer)


def test_tanimoto_accepts_smiles():
    """The model takes identity-space SMILES and featurizes internally."""
    model = TanimotoGPModel()
    model.initialize(SMILES, MOL_Y)
    model.fit()

    # train_X is stored as feature tensors, not SMILES
    assert torch.is_tensor(model.train_X)
    assert model.train_X.shape[0] == len(SMILES)

    mean, std = model(SMILES)
    assert mean.shape == (len(SMILES), 1)
    assert std.shape == (len(SMILES), 1)


def test_tanimoto_update_with_smiles():
    model = TanimotoGPModel()
    model.initialize(SMILES, MOL_Y)
    model.fit()
    model.update(SMILES_NEW, MOL_Y_NEW)

    assert model.train_X.shape[0] == len(SMILES) + len(SMILES_NEW)
    assert model.train_y.shape[0] == len(SMILES) + len(SMILES_NEW)
    model.fit()


# --- fit guard ------------------------------------------------------------


def test_predict_before_fit_raises():
    model = GPModel()
    model.initialize(CONT_X, CONT_Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        model(CONT_X)


def test_loss_before_fit_raises():
    model = GPModel()
    model.initialize(CONT_X, CONT_Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.loss()
