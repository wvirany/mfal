from abc import ABC, abstractmethod

import torch


class SurrogateModel(ABC):
    """Base class for surrogate models"""

    @abstractmethod
    def initialize(self, train_X, train_y):
        pass

    @abstractmethod
    def fit(self):
        """
        Fit model to training data.

        Args:
            train_X: Training inputs (B, N, d)
            train_y: Training targets (B, N, m)
        """
        pass

    @abstractmethod
    def update(self, new_X: torch.Tensor, new_y: torch.Tensor, state_dict=None):
        """
        Update model with new training data.

        Args:
            new_X: New data point(s) to add (B, N, d)
            new_y: New target(s) (B, N, m)
        """
        pass
