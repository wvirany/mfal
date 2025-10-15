import os

import numpy as np
import pytest

from mfal.data import load_mcl1_data
from mfal.utils.al_loop import initialize_centroid
from mfal.utils.embeddings import get_or_generate_embeddings


@pytest.fixture
def sample_smiles():
    """Load a small sample of SMILES for testing."""
    df = load_mcl1_data()
    return df["prot_smiles"].tolist()[:10]


@pytest.fixture
def test_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""

    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_morgan_fp_generation(sample_smiles, test_cache_dir):
    """Test Morgan FP generation."""
    embeddings = get_or_generate_embeddings(sample_smiles, "morgan_fp", cache_dir=test_cache_dir)

    assert embeddings.shape == (len(sample_smiles), 2048)
    assert not np.all(embeddings == 0), "All embeddings are zero"


def test_caching_works(sample_smiles, test_cache_dir):
    """Test that caching works."""

    embeddings1 = get_or_generate_embeddings(sample_smiles, "morgan_fp", cache_dir=test_cache_dir)

    # Should load from cache
    embeddings2 = get_or_generate_embeddings(sample_smiles, "morgan_fp", cache_dir=test_cache_dir)

    # Check cache file exists
    cache_file = os.path.join(test_cache_dir, "morgan_fp.npz")
    assert os.path.exists(cache_file), "Cache file not created"

    # Check embeddings are identical
    assert np.allclose(embeddings1, embeddings2), "Cached embeddings don't match original"


def test_centroid_initialization(sample_smiles, test_cache_dir):
    """Test centroid initialization."""
    embeddings = get_or_generate_embeddings(sample_smiles, "morgan_fp", cache_dir=test_cache_dir)

    centroid_idx = initialize_centroid(embeddings)
    centroid = embeddings[centroid_idx]

    assert centroid.shape == (2048,)
    assert not np.all(centroid == 0), "Centroid is all zeros"


def test_full_dataset_generation():
    """Test embedding generation for full dataset."""
    df = load_mcl1_data()
    smiles = df["prot_smiles"].tolist()

    embeddings = get_or_generate_embeddings(smiles, "morgan_fp")

    assert embeddings.shape[0] == len(
        smiles
    ), "Number of embeddings does not match number of molecules"
    assert embeddings.shape[1] == 2048, "Embedding dimension should be 2048"
    assert not np.all(embeddings == 0), "All embeddings are zero"

    # Test that we can find centroid
    centroid_idx = initialize_centroid(embeddings)
    assert 0 <= centroid_idx < len(smiles), "Initialization index out of bounds"
