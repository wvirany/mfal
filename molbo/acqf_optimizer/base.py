import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from molbo.acquisition import Acquisition


@dataclass
class Initialization:
    train_X: Any
    train_y: torch.Tensor
    observed_indices: Optional[List[int]] = None


@dataclass
class OptimizationResult:
    new_X: Any
    acq_val: torch.Tensor
    observed_indices: Optional[List[int]] = None
    optimization_time: Optional[float] = None


class AcqfOptimizer(ABC):

    def optimize(
        self, acq_func: Acquisition, candidates: Any = None, observed_indices: List[int] = None
    ) -> OptimizationResult:
        start = time.perf_counter()
        result = self._optimize(acq_func, candidates, observed_indices)
        result.optimization_time = time.perf_counter() - start
        return result

    @abstractmethod
    def _optimize(
        self, acq_func: Acquisition, candidates: Any = None, observed_indices: List[int] = None
    ) -> tuple:
        pass

    @abstractmethod
    def sample_init(self, oracle, n_init: int) -> Initialization:
        pass
