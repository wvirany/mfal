import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from molbo.acquisition import Acquisition


@dataclass
class Initialization:
    train_X: torch.Tensor
    train_y: torch.Tensor
    observed_indices: Optional[List[int]] = None
    smiles: Optional[List[str]] = None


@dataclass
class OptimizationResult:
    new_X: torch.Tensor
    acq_val: torch.Tensor
    smiles: Optional[List[str]] = None
    optimization_time: Optional[float] = None


class AcqfOptimizer(ABC):

    def optimize(self, acq_func: Acquisition, candidates: torch.Tensor = None) -> tuple:
        start = time.perf_counter()
        result = self._optimize(acq_func, candidates)
        result.optimization_time = time.perf_counter() - start
        return result

    @abstractmethod
    def _optimize(self, acq_func: Acquisition, candidates: torch.Tensor = None) -> tuple:
        pass

    @abstractmethod
    def sample_init(self, oracle, n_init: int) -> Initialization:
        pass
