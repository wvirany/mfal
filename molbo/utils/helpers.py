from typing import List

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.QED import qed


def sample_init(oracle, n_init, candidates=None):

    if candidates is not None:
        indices = np.random.choice(len(candidates), n_init, replace=False).tolist()
        X, y = oracle[indices]
    else:
        indices = None
        bounds = oracle.bounds
        X = torch.rand(n_init, oracle.dim).to(bounds) * (bounds[1] - bounds[0]) + bounds[0]
        y = oracle(X)

    return X, y, indices


# Compute ECFP4 fingerprints:
def smiles_to_morgan_fp(smiles: str, as_tensor: bool = True, radius: int = 2, fp_size: int = 2048):
    # Return fingerprint as numpy array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    if as_tensor:
        return torch.from_numpy(fp_gen.GetCountFingerprintAsNumPy(mol)).to(torch.float64)

    return fp_gen.GetCountFingerprint(mol)


def smiles_to_qed(smiles: str | List[str]) -> torch.Tensor:
    if isinstance(smiles, str):
        smiles = [smiles]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    scores = [qed(mol) for mol in mols]
    return torch.tensor(scores, dtype=torch.float64)


def get_centroid_indices(smiles_list, scores, tanimoto_threshold=0.7):
    fps = [smiles_to_morgan_fp(s, as_tensor=False) for s in smiles_list]
    sorted_indices = torch.argsort(scores, descending=True)
    centroids = []  # list of (fp, idx)

    for i in sorted_indices:
        fp = fps[i]
        if len(centroids) == 0:
            centroids.append((fp, i.item()))
        else:
            sims = [DataStructs.TanimotoSimilarity(fp, c_fp) for c_fp, _ in centroids]
            if max(sims) < tanimoto_threshold:
                centroids.append((fp, i.item()))

    return [idx for _, idx in centroids]


def get_centroid_indices_from_fps(
    fps: torch.Tensor,
    scores: torch.Tensor,
    tanimoto_threshold: float = 0.7,
    max_centroids: int = None,
):
    """Greedy Tanimoto clustering on fingerprint tensors in descending score order."""
    sorted_indices = torch.argsort(scores, descending=True)
    centroid_fps = []
    centroid_indices = []

    for i in sorted_indices:
        fp = fps[i]
        if len(centroid_fps) == 0:
            centroid_fps.append(fp)
            centroid_indices.append(i.item())
        else:
            centroids = torch.stack(centroid_fps)
            dot = (fp * centroids).sum(dim=1)
            denom = fp.pow(2).sum() + centroids.pow(2).sum(dim=1) - dot
            sims = dot / denom.clamp(min=1e-10)
            if sims.max().item() < tanimoto_threshold:
                centroid_fps.append(fp)
                centroid_indices.append(i.item())

        if max_centroids is not None and len(centroid_indices) == max_centroids:
            break

    return centroid_indices
