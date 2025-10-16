from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings"


# Compute ECFP4 fingerprints:
def smiles_to_morgan_fp(smiles: str, as_numpy: bool = True, radius: int = 2, fp_size: int = 2048):
    # Return fingerprint as numpy array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    if as_numpy:
        return fp_gen.GetCountFingerprintAsNumPy(mol)

    return fp_gen.GetCountFingerprint(mol)


def generate_morgan_fps(
    smiles_list: List[str], radius: int = 2, fp_size: int = 2048, as_numpy: bool = True
):
    fps = []

    print(f"Generating Morgan fingerprints for {len(smiles_list)} molecules...")
    for smiles in tqdm(smiles_list):
        try:
            fp = smiles_to_morgan_fp(smiles, as_numpy=as_numpy, radius=radius, fp_size=fp_size)
            fps.append(fp)
        except ValueError as e:
            print(e)
            fps.append(None)  # Append None for invalid SMILES

    fps = np.array(fps)

    return fps


def save_embeddings(embeddings: np.ndarray, filename: str):
    """Save embeddings to disk for caching."""
    np.savez_compressed(filename, embeddings=embeddings)
    print(f"Embeddings saved to {filename}")


def load_embeddings(filename: str):
    """Load embeddings from disk."""
    return np.load(filename)["embeddings"]


def get_embeddings(
    smiles_list: List[str],
    embedding_type: str = "morgan_fp",
    cache_dir: str = None,
    **kwargs,
):
    """Get or generate embeddings."""

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{embedding_type}.npz"

    # Try to load from cache
    if cache_path.exists():
        return load_embeddings(cache_path)

    # Generate embeddings
    if embedding_type == "morgan_fp":
        radius = kwargs.get("radius", 2)
        fp_size = kwargs.get("fp_size", 2048)
        embeddings = generate_morgan_fps(smiles_list, as_numpy=True, radius=radius, fp_size=fp_size)
    else:
        raise ValueError(f"Invalid embedding type: {embedding_type}")

    # Save to cache
    save_embeddings(embeddings, cache_path)

    return embeddings
