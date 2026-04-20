from pathlib import Path
from typing import List

import pandas as pd
import torch

from molbo.dataset.base import Dataset

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tiny_lib"


class TinyLibraryDataset(Dataset):

    def __init__(self, path: str = DATA_DIR / "properties_30k.csv"):
        df = pd.read_csv(path)
        self._smiles = df["smiles"].tolist()
        self._columns = {"qed": torch.tensor(df["qed"].values, dtype=torch.float64)}

    @property
    def smiles(self) -> List[str]:
        return self._smiles

    @property
    def columns(self):
        return self._columns

    @property
    def _candidates_path(self) -> Path:
        return DATA_DIR / "candidates.pt"

    @property
    def _path(self) -> Path:
        return DATA_DIR / "properties_30k.csv"
