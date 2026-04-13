from pathlib import Path
from typing import List

from molbo.dataset.base import Dataset

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TinyLibraryDataset(Dataset):

    def __init__(self, path: str = DATA_DIR / "tiny_lib/smiles_30k.txt"):
        with open(path) as f:
            self._smiles = [line.strip() for line in f.readlines()]

    @property
    def smiles(self) -> List[str]:
        return self._smiles
