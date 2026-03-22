import numpy as np
import torch

from molbo.oracle import AnalyticOracle, LookupOracle


def sample_init(oracle, n_init):
    if isinstance(oracle, AnalyticOracle):
        indices = None
        bounds = oracle.bounds
        X = torch.rand(n_init, oracle.dim) * (bounds[1] - bounds[0]) + bounds[0]
    elif isinstance(oracle, LookupOracle):
        indices = np.random.choice(len(oracle.X_data), n_init, replace=False)
        X = oracle.X_data[indices]

    return X, oracle(X), indices
