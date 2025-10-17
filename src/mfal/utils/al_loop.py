import numpy as np
import torch
import wandb
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
    batch_size: int = 100,
    verbose: bool = True,
    wandb_run: wandb.Run = None,
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

    # MinMax scaling of embeddings
    emb_min = embeddings.min(axis=0, keepdims=True)
    emb_max = embeddings.max(axis=0, keepdims=True)
    embeddings = (embeddings - emb_min) / (emb_max - emb_min + 1e-8)  # Avoid division by zero

    # Initialize with centroid
    if initial_idx is None:
        # initial_idx = initialize_centroid(embeddings)
        initial_idx = np.random.randint(0, len(embeddings))

    # Initialize queried indices
    queried_indices = [initial_idx]
    queried_scores = [oracle_fn(initial_idx)]

    # Prepare training data
    train_x = torch.tensor(embeddings[queried_indices], dtype=torch.float64, device=device)
    train_y = torch.tensor([[-queried_scores[0]]], dtype=torch.float64, device=device)

    # Keep embeddings on CPU
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float64)

    # Initialize model
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
    print(f"Batch size: {batch_size}")
    print(f"Initial: idx={initial_idx}, score={queried_scores[0]:.3f}")
    print(f"{'='*60}\n")

    for iteration in tqdm(range(n_iterations), desc="AL progress"):

        # Fit GP hyperparameters
        fit_gpytorch_mll(mll)
        model.eval()

        # Select next query
        next_idx = select_next_molecule(
            model, train_y, embeddings_tensor, queried_indices, batch_size
        )

        # Evaluate oracle
        next_score = oracle_fn(next_idx)
        queried_indices.append(next_idx)
        queried_scores.append(next_score)

        # Update training data
        new_x = torch.tensor(embeddings[next_idx], dtype=torch.float64, device=device).unsqueeze(0)
        new_y = torch.tensor([[-next_score]], dtype=torch.float64, device=device)
        train_x = torch.cat([train_x, new_x], dim=0)  # (n+1, d)
        train_y = torch.cat([train_y, new_y], dim=0)  # (n+1, 1)

        # Update model
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        # Compute metrics
        found_top1 = set(queried_indices) & top1_indices
        retrieval = len(found_top1) / len(top1_indices) * 100
        best_score = min(queried_scores)

        top1_retrieval.append(retrieval)
        best_scores.append(best_score)

        # Log to wandb
        if wandb_run is not None:
            wandb.log(
                {
                    "iteration": iteration,
                    "top1_retrieval": retrieval,
                    "best_score": best_score,
                    "current_score": next_score,
                    "n_top1_found": len(found_top1),
                },
                step=iteration + 1,
            )

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
