from abc import ABC, abstractmethod
from typing import Set

import numpy as np

from mfal.models.base import SurrogateModel


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def select_next(
        self,
        model: SurrogateModel,
        embeddings: np.ndarray,
        queried_indices: Set[int],
        current_best: float,
    ) -> int:
        """
        Select next point to query.

        Args:
            model: Trained surrogate model
            embeddings: All candidate embeddings (N, n_features)
            queried_indices: Set of observed indices
            current_best: Best score observed so far (necessary for EI)

        Returns:
            next_idx: Index of next point to query
        """
        pass
