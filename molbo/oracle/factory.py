import torch

from molbo.dataset.base import Dataset
from molbo.oracle.base import Oracle
from molbo.oracle.lookup import LookupOracle
from molbo.utils.helpers import smiles_to_morgan_fp


def oracle_from_dataset(
    dataset: Dataset,
    column: str,
    oracle: Oracle = None,
    noise_std: float = 0.0,
    negate: bool = False,
    n: int = None,
) -> LookupOracle:
    X = dataset.candidates[:n] if n is not None else dataset.candidates

    if column in dataset.columns:
        y = dataset.columns[column][:n].unsqueeze(-1)
    else:
        assert oracle is not None, f"Column '{column}' not in dataset and no oracle provided"
        full_X = dataset.candidates
        y_full = oracle(full_X)
        dataset.save_column(column, y_full.squeeze(-1))
        y = y_full[:n].unsqueeze(-1) if n is not None else y_full

    if negate:
        y = -y

    return LookupOracle(X_data=X, y_data=y, noise_std=noise_std)
