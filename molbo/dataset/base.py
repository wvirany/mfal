from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from molbo.utils import smiles_to_morgan_fp


class Dataset(ABC):

    @property
    @abstractmethod
    def smiles(self) -> List[str]: ...

    @property
    @abstractmethod
    def _candidates_path(self) -> Path: ...

    @property
    def columns(self) -> Dict[str, torch.Tensor]:
        return {}

    @property
    def candidates(self) -> torch.Tensor:
        if self._candidates_path.exists():
            return torch.load(self._candidates_path)
        candidates = torch.vstack([smiles_to_morgan_fp(s) for s in self.smiles])
        torch.save(candidates, self._candidates_path)
        return candidates

    def save_column(self, name: str, values: torch.Tensor):
        df = pd.read_csv(self._path)
        df[name] = values.numpy()
        df.to_csv(self._path, index=False)
        self._columns[name] = values
