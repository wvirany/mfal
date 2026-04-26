from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from molbo.dataset.base import Dataset

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zinc250k"


class Zinc250kDataset(Dataset):

    def __init__(self, path: str = DATA_DIR / "results.csv"):
        df = pd.read_csv(path)
        self._smiles = df["smiles"].tolist()
        self._columns = {
            "qed": torch.tensor(df["qed"].values, dtype=torch.float64),
            "jnk3": torch.tensor(df["jnk3"].values, dtype=torch.float64),
            "gsk3b": torch.tensor(df["gsk3b"].values, dtype=torch.float64),
        }

        super().__init__()

        self._columns = {k: v[self._unique_indices] for k, v in self._columns.items()}

    @property
    def smiles(self) -> List[str]:
        return self._smiles

    @property
    def _candidates_path(self) -> Path:
        return DATA_DIR / "candidates.pt"

    @property
    def columns(self) -> Dict[str, torch.Tensor]:
        return self._columns

    @property
    def _path(self) -> Path:
        return DATA_DIR / "results.csv"
