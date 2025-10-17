from typing import Set

import numpy as np

from mfal.acquisition.base import AcquisitionFunction
from mfal.models.base import SurrogateModel


class RandomAcquisition(AcquisitionFunction):
    """
    Random acquisition function.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random acquisition function.

        Args:
            random_state: Random seed
        """
        self.rng = np.random.RandomState(random_state)

    def select_next(
        self,
        model: SurrogateModel,
        embeddings: np.ndarray,
        queried_indices: Set[int],
        current_best: float,
    ) -> int:
        """
        Select random unqueried point.

        Args:
            model: Ignored
            embeddings: All candidate embeddings (N, n_features)
            queried_indices: Set of observed indices
            current_best: Best score observed so far (ignored)

        Returns:
            next_idx: Randomly selected index of next point to query
        """

        # Get all unqueried indices
        all_indices = set(range(len(embeddings)))
        unqueried_indices = list(all_indices - set(queried_indices))

        if len(unqueried_indices) == 0:
            raise ValueError("No unqueried points left")

        # Random selection
        return self.rng.choice(unqueried_indices)
