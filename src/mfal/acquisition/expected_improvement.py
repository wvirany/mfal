from typing import Set

import numpy as np
from scipy.stats import norm

from mfal.acquisition.base import AcquisitionFunction
from mfal.models.base import SurrogateModel


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected improvement acquisition function.

    Works with any surrogate model that provides predict(X) -> (mean, std).
    """

    def __init__(self, beta: float = 0.01):
        """
        Initialize EI.

        Args:
            beta: Exploration-exploitation trade-off parameter
        """
        self.beta = beta

    def select_next(
        self,
        model: SurrogateModel,
        embeddings: np.ndarray,
        queried_indices: Set[int],
        current_best: float,
    ) -> int:
        """
        Select point with highest EI.

        Args:
            model: Trained surrogate model
            embeddings: All molecule embeddings (N, n_features)
            queried_indices: Set of already-queried indices
            current_best: Best score observed so far

        Returns:
            next_idx: Index of point with highest EI
        """

        # Get unqueried indices
        all_indices = set(range(len(embeddings)))
        unqueried_indices = list(all_indices - queried_indices)

        if len(unqueried_indices) == 0:
            raise ValueError("No unqueried indices left")

        # Get predictions for unqueried points
        X_unqueried = embeddings[unqueried_indices]
        mean, std = model.predict(X_unqueried)

        # Compute EI
        ei_values = self._compute_ei(mean, std, current_best)

        # Select point with highest EI
        best_idx = np.argmax(ei_values)
        return unqueried_indices[best_idx]

    def _compute_ei(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """
        Compute EI for each point.
        """

        # Avoid numerical issues with small std
        std = np.maximum(std, 1e-9)

        # Compute improvement
        improvement = mean - f_best - self.beta

        # Standardize
        Z = improvement / std

        # Compute the expected improvement
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)

        # If std is very small, set EI to 0
        ei[std < 1e-9] = 0.0

        return ei
