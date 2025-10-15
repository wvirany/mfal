import numpy as np


def initialize_centroid(embeddings: np.ndarray):
    """Find closest molecule to centroid based on Euclidean distance."""
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argmin(distances)
