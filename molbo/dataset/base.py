from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class Dataset(ABC):

    @property
    @abstractmethod
    def smiles(self) -> List[str]: ...

    @property
    def columns(self) -> Dict[str, torch.Tensor]:
        return {}
