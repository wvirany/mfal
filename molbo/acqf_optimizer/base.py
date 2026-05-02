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
    smiles: Optional[List[str]] = None  # Used for MolecularOracle
    all_acq_values: Optional[torch.Tensor] = None  # Returned by PoolSampler for metrics


class AcqfOptimizer(ABC):

    @abstractmethod
    def optimize(self, acq_func: Acquisition, candidates: torch.Tensor = None) -> tuple:
        pass

    @abstractmethod
    def sample_init(self, oracle, n_init: int) -> Initialization:
        pass
