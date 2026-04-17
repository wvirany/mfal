import torch

from molbo.dataset.base import Dataset
from molbo.oracle.lookup import LookupOracle
from molbo.utils.helpers import smiles_to_morgan_fp


def oracle_from_dataset(
    dataset: Dataset,
    column: str = None,
    evaluate_fn=None,
    noise_std: float = 0.0,
    negate: bool = False,
    n: int = None,
) -> LookupOracle:
    assert (column is None) != (
        evaluate_fn is None
    ), "Exactly one of column or evaluate_fn must be provided"

    smiles = dataset.smiles[:n] if n is not None else dataset.smiles

    X = torch.vstack([smiles_to_morgan_fp(s) for s in smiles])

    if column is not None:
        y = dataset.columns[column][:n].unsqueeze(-1)
    else:
        y = torch.tensor(evaluate_fn(smiles), dtype=torch.float64).unsqueeze(-1)

    if negate:
        y = -y

    return LookupOracle(X_data=X, y_data=y, noise_std=noise_std)
