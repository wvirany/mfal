from abc import ABC, abstractmethod

import torch

from molbo.acquisition import Acquisition


class AcqfOptimizer(ABC):

    @abstractmethod
    def optimize(self, acq_func: Acquisition, candidates: torch.Tensor = None) -> tuple:
        """
        Optimize acquisition function and return next candidate(s).

        Returns:
            new_X: (q, d) tensor
            acq_val: (q,) tensor
        """
        pass
