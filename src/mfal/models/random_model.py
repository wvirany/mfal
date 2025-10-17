import numpy as np

from mfal.models.base import SurrogateModel


class RandomModel(SurrogateModel):
    """
    Dummy model for random baseline. Just a placeholder for use with RandomAcquisition.
    """

    def __init__(self, random_state: int = 42):
        """Initialize random model."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        No-op fit for random model.
        """
        pass

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return placeholders for mean and std.
        """
        n = len(X)
        mean = np.zeros(n)
        std = np.ones(n)
        return mean, std

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        No-op update for random model.
        """
        pass
