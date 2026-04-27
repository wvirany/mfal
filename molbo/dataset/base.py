from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from molbo.utils import smiles_to_morgan_fp


class Dataset(ABC):

    def __init__(self):

        # Load cached candidates or compute
        if self._candidates_path.exists():
            candidates = torch.load(self._candidates_path)
        else:
            candidates = torch.vstack([smiles_to_morgan_fp(s) for s in self.smiles])
            torch.save(candidates, self._candidates_path)

        # Deduplicate candidates at representation level
        seen = set()
        unique_indices = []
        for i, row in enumerate(candidates):
            key = row.numpy().tobytes()
            if key not in seen:
                seen.add(key)
                unique_indices.append(i)
        self._unique_indices = torch.tensor(unique_indices)
        self._candidates = candidates[self._unique_indices]

    @property
    @abstractmethod
    def smiles(self) -> List[str]: ...

    @property
    def candidate_smiles(self) -> List[str]:
        return [self.smiles[i] for i in self._unique_indices]

    @property
    @abstractmethod
    def _candidates_path(self) -> Path: ...

    @property
    def candidates(self) -> torch.Tensor:
        return self._candidates

    @property
    def columns(self) -> Dict[str, torch.Tensor]:
        return {}

    def save_column(self, name: str, values: torch.Tensor):
        df = pd.read_csv(self._path)
        df[name] = values.numpy()
        df.to_csv(self._path, index=False)
        self._columns[name] = values
