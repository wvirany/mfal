from abc import abstractmethod
from typing import List, Optional

import torch
from rdkit.Chem import QED, MolFromSmiles

from molbo.oracle.base import Oracle


class MolecularOracle(Oracle):
    """
    Oracle that evaluates molecules directly from SMILES strings.
    """

    def __init__(self, noise_std: float = 0.0, optimal_value: Optional[float] = None):
        super().__init__(noise_std)
        self._optimal_value = optimal_value

    @abstractmethod
    def _evaluate(self, smiles: List[str]) -> torch.Tensor:
        """
        Args:
            smiles: List of N SMILES strings
        Returns:
            y: (N, 1) float64 tensor
        """
        pass

    @property
    def optimal_value(self):
        return self._optimal_value

    def to(self, device):
        return self


class QEDOracle(MolecularOracle):

    def __init__(self, noise_std: float = 0.0):
        super().__init__(noise_std=noise_std, optimal_value=1.0)

    def _evaluate(self, smiles: List[str]) -> torch.Tensor:
        scores = []
        for smi in smiles:
            mol = MolFromSmiles(smi)
            scores.append(QED.qed(mol) if mol is not None else 0.0)
        return torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)
