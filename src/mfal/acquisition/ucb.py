from typing import Set

import numpy as np

from mfal.acquisition.base import AcquisitionFunction
from mfal.models.base import SurrogateModel


class UpperConfidenceBound(AcquisitionFunction):
    """UCB acquisition function."""

    def __init__(self, beta: float = 1.0):
        """
        Initialize UCB.

        Args:
            beta: Exploration-exploitation trade-off parameter
        """
        self.beta = beta

    def select_next(
        self,
        model: SurrogateModel,
        embeddings: np.ndarray,
        queried_indices: Set[int],
        current_best: float = None,
    ):
        """
        Select point with highest UCB.

        Args:
            model: Trained surrogate model
            embeddings: All molecule embeddings (N, n_features)
            queried_indices: Set of already-queried indices
            current_best: Dummy parameter, used for EI

        Returns:
            next_idx: Index of point with highest UCB
        """

        # Get unqueried indices
        all_indices = set(range(len(embeddings)))
        unqueried_indices = list(all_indices - queried_indices)

        if len(unqueried_indices) == 0:
            raise ValueError("No unqueried indices left")

        # Get predictions for unqueried points
        X_unqueried = embeddings[unqueried_indices]
        mean, std = model.predict(X_unqueried)

        ucb_values = mean + self.beta * std

        # Select point with highest UCB
        best_idx = np.argmax(ucb_values)
        return unqueried_indices[best_idx]

    def get_beta(self) -> float:
        return self.beta

    def set_beta(self, beta: float):
        self.beta = beta
