import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from tqdm import tqdm

from mfal.models.gp import initialize_gp
from mfal.utils.acquisition import select_next_molecule


def initialize_centroid(embeddings: np.ndarray):
    """Find closest molecule to centroid based on Euclidean distance."""
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argmin(distances)


def run_al_loop(
    embeddings: np.ndarray,  # (N, d) numpy array
    oracle_fn,  # Function: idx -> score
    top1_indices,  # Set of ground truth top 1% indices
    n_iterations: int = 1000,
    initial_idx: int = None,
    score_type: str = "docking",
    random_seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
):
    """
    Run active learning loop.
    """

    # Set seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device}")
    device = torch.device(device)
    print(f"Using device: {device}")

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float64, device=device)

    # Initialize with centroid
    if initial_idx is None:
        initial_idx = initialize_centroid(embeddings)

    # Initialize queried indices
    queried_indices = [initial_idx]
    queried_scores = [oracle_fn(initial_idx)]

    # Prepare training data
    train_x = embeddings_tensor[queried_indices]

    # Negate scores: maximizing negative binding affinity (since lower binding affinity is better)
    train_y = torch.tensor([[-queried_scores[0]]], dtype=torch.float64, device=device)

    # Initialize model (will use same device as train_x)
    model, mll = initialize_gp(train_x, train_y)

    # Tracking metrics
    top1_retrieval = []
    best_scores = []

    # Run active learning loop
    print(f"\n{'='*60}")
    print("Running active learning loop")
    print(f"Dataset size: {len(embeddings)}")
    print(f"Iterations: {n_iterations}")
    print(f"Score type: {score_type}")
    print(f"Initial: idx={initial_idx}, score={queried_scores[0]:.3f}")
    print(f"{'='*60}\n")

    for iteration in tqdm(range(n_iterations), desc="AL progress"):
        # Fit GP hyperparameters
        fit_gpytorch_mll(mll)

        # Select next query
        next_idx = select_next_molecule(model, train_y, embeddings_tensor, queried_indices)

        # Evaluate oracle
        next_score = oracle_fn(next_idx)
        queried_indices.append(next_idx)
        queried_scores.append(next_score)

        # Update training data
        new_x = embeddings_tensor[next_idx].unsqueeze(0)  # (1, d)
        new_y = torch.tensor([[-next_score]], dtype=torch.float64, device=device)
        train_x = torch.cat([train_x, new_x], dim=0)  # (n+1, d)
        train_y = torch.cat([train_y, new_y], dim=0)  # (n+1, 1)

        # Update model
        model, mll = initialize_gp(train_x, train_y, model.state_dict())

        # Compute metrics
        found_top1 = set(queried_indices) & top1_indices
        retrieval = len(found_top1) / len(top1_indices) * 100
        best_score = min(queried_scores)

        top1_retrieval.append(retrieval)
        best_scores.append(best_score)

        if verbose and (iteration % 100 == 0):
            print(f"\n--- Iteration {iteration+1} ---")
            print(f"  Top-1% retrieval: {retrieval:.2f}% ({len(found_top1)}/{len(top1_indices)})")
            print(f"  Best score: {best_score:.3f}")
            print(f"  Latest score: {next_score:.3f}")

    # Final summary
    print(f"\n{'='*60}")
    print("Active learning complete!")
    print(f"{'='*60}")
    print(f"Top-1% retrieval: {top1_retrieval[-1]:.2f}%")
    print(f"Best score: {best_scores[-1]:.3f}")

    return {
        "queried_indices": queried_indices,
        "queried_scores": queried_scores,
        "top1_retrieval": top1_retrieval,
        "best_scores": best_scores,
    }
