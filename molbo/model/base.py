from abc import ABC, abstractmethod
from typing import Any

import torch


class SurrogateModel(ABC):
    """Base class for surrogate models."""

    @abstractmethod
    def initialize(self, train_X: Any, train_y: torch.Tensor) -> None:
        """Initialize model with training data. Called once before the BO loop."""
        pass

    @abstractmethod
    def fit(self) -> None:
        """Fit model to current training data."""
        pass

    @abstractmethod
    def update(self, new_X: Any, new_y: torch.Tensor) -> None:
        """Append new observations and reinitialize the model.

        Args:
            new_X: New candidate representation(s) - type matches that of
                    the current run (e.g. tensor, SMILES list).
            new_y: New target value(s), shape (q, 1).
        """
        pass

    @abstractmethod
    def __call__(self, X: Any):
        """Return (mean, stddev) posterior predictions for X."""
        pass

    @abstractmethod
    def loss(self) -> torch.Tensor:
        """Return marginal log likelihood on current training data."""
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        pass

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fit. Call fit() before predicting.")
