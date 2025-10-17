from abc import ABC, abstractmethod

import numpy as np


class SurrogateModel(ABC):
    """Base class for surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model to training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for test points.

        Args:
            X: Test features (n_test, n_features)

        Returns:
            mean: Predicted means (n_test,)
            std: Predicted standard deviations (n_test,)
        """
        pass

    @abstractmethod
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update model with new training data.

        This can be implemented as:
        - Incremental updates (if model supports it)
        - Full refit (concatenate with existing data and call fit())
        - No-op (for models that don't need updates)

        Args:
            X_new: New data point(s) to add (n_new, n_features)
            y_new: New target(s) (n_new,)
        """
        pass
