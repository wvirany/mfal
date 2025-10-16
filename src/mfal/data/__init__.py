from pathlib import Path

import numpy as np
import pandas as pd


def load_mcl1_data():
    df = pd.read_csv(Path(__file__).parent / "results.csv")
    return df


def compute_top_k_indices(scores, k_percent=1.0):
    """
    Get indices of top k% molecules.

    Args:
        scores: Array of scores (lower is better)
        k_percent: Percentile (default 1.0 for top 1%)

    Returns:
        top_indices: Set of indices of top k% molecules
        threshold: Score threshold for top k% molecules
    """
    k = int(len(scores) * k_percent / 100)

    # For binding scores, lower is better
    top_indices = set(np.argsort(scores)[:k])
    threshold = np.sort(scores)[k]

    return set(top_indices), threshold


def get_oracle_function(df, score_type: str = "mmgbsa"):
    """
    Create oracle function for AL queries.

    Args:
        df: DataFrame with scores
        score_type: 'docking' or 'mmgbsa'

    Returns:
        oracle_fn: Function that takes index, returns score
    """

    if score_type == "docking":
        scores = df["vina_score"].values
    elif score_type == "mmgbsa":
        scores = df["mmgbsa_score"].values
    else:
        raise ValueError(f"Invalid score type: {score_type}")

    def oracle_fn(idx):
        """Query oracle for score at index idx."""
        return scores[idx]

    return oracle_fn, scores
