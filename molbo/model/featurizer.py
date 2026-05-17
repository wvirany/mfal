from abc import ABC, abstractmethod
from typing import List

import torch

from molbo.utils.helpers import smiles_to_morgan_fp


class Featurizer(ABC):
    """Maps a candidate representation to the tensor a model's kernel consumes.

    A model owns a Featurizer instance and applies it internally. This keeps the
    rest of the BO stack (History, BOLoop, acquisition optimizers) agnostic to
    the candidate representation - they pass identity-space X around, and only
    the model turns it into features.
    """

    @abstractmethod
    def __call__(self, X) -> torch.Tensor:
        """Featurize a batch of candidates into an ``(n, d)`` tensor."""
        ...


class IdentityFeaturizer(Featurizer):
    """Pass-through featurizer for domains where X is already a feature tensor.

    Used by continuous-domain models - the candidate representation and the
    kernel input are the same thing.
    """

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X


class MorganFingerprintFeaturizer(Featurizer):
    """Featurizes SMILES strings into Morgan count fingerprints.

    Args:
        radius: Morgan radius (default 2, i.e. ECFP4).
        fp_size: Fingerprint length in bits.
    """

    def __init__(self, radius: int = 2, fp_size: int = 2048):
        self.radius = radius
        self.fp_size = fp_size

    def __call__(self, X: List[str]) -> torch.Tensor:
        return torch.stack(
            [smiles_to_morgan_fp(smiles, radius=self.radius, fp_size=self.fp_size) for smiles in X]
        )
