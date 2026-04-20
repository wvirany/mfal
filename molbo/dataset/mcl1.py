from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from molbo.dataset.base import Dataset

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mcl1"


class Mcl1Dataset(Dataset):

    def __init__(self, path: str = DATA_DIR / "results.csv"):
        df = pd.read_csv(path)
        self._smiles = df["prot_smiles"].tolist()
        self._columns = {
            "qed": torch.tensor(df["qed_score"].values, dtype=torch.float64),
            "vina": torch.tensor(df["vina_score"].values, dtype=torch.float64),
            "mmgbsa": torch.tensor(df["mmgbsa_score"].values, dtype=torch.float64),
        }

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
