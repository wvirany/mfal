from typing import Callable, List

import numpy as np
from tqdm import tqdm

import wandb
from mfal.acquisition.base import AcquisitionFunction
from mfal.models.base import SurrogateModel


def initialize_centroid(embeddings: np.ndarray):
    """Find closest molecule to centroid based on Euclidean distance."""
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argmin(distances)


def run_al_loop(
    embeddings: np.ndarray,
    oracle_fn: Callable,
    top1_indices: List[int],
    model: SurrogateModel,
    acq_func: AcquisitionFunction,
    n_iterations: int = 1000,
    n_initial: int = 100,
    score_type: str = "docking",
    random_seed: int = 42,
    verbose: bool = True,
    wandb_run: wandb.Run = None,
):
    """
    Run active learning loop.

    Currently only supports one-at-a-time querying.

    Args:
        embeddings: (N, d) numpy array of embeddings
        oracle_fn: Function to evaluate oracle score
        top1_indices: List of top 1% indices
        model: Surrogate model
        acq_func: Acquisition function
        n_iterations: Number of iterations (note we run for n_iterations - n_initial iterations)
        n_initial: Number of initial points
        score_type: Score type
        random_seed: Random seed
    """

    # Set seed
    np.random.seed(random_seed)

    # Normalize embeddings
    emb_min = embeddings.min(axis=0, keepdims=True)
    emb_max = embeddings.max(axis=0, keepdims=True)
    embeddings_norm = (embeddings - emb_min) / (emb_max - emb_min + 1e-8)

    # Initialize with random sampling
    initial_indices = np.random.choice(
        len(embeddings),
        size=n_initial,
        replace=False,
    )
    queried_indices = initial_indices.tolist()
    queried_scores = [oracle_fn(idx) for idx in initial_indices]

    # Initial model fit
    X_train = embeddings_norm[queried_indices]
    y_train = -np.array(queried_scores)  # Negate for maximization
    model.fit(X_train, y_train)

    # Initialize UCB parameter
    if isinstance(acq_func, AcquisitionFunction):
        acq_func.set_beta(beta=10.0)

    # Tracking metrics
    top1_retrieval = []
    best_scores = []

    # Print header
    print(f"\n{'='*60}")
    print("Running active learning loop")
    print(f"Model: {model.__class__.__name__}")
    print(f"Acquisition: {acq_func.__class__.__name__}")
    print(f"Dataset size: {len(embeddings_norm)}")
    print(f"Iterations: {n_iterations}")
    print(f"Score type: {score_type}")
    print(f"Initial: n={n_initial}, best={min(queried_scores):.3f}")
    print(f"{'='*60}\n")

    # Main AL loop
    for iteration in tqdm(range(n_iterations - n_initial), desc="AL progress"):

        # UCB beta schedule
        if isinstance(acq_func, AcquisitionFunction):
            b = 10.0 / (iteration + 1)
            acq_func.set_beta(beta=b)

        # Select next query
        next_idx = acq_func.select_next(
            model=model,
            embeddings=embeddings_norm,
            queried_indices=set(queried_indices),
            current_best=-min(queried_scores),
        )

        # Evaluate oracle
        next_score = oracle_fn(next_idx)
        queried_indices.append(next_idx)
        queried_scores.append(next_score)

        # Update model with new training data
        X_new = embeddings_norm[next_idx].reshape(1, -1)  # (d,) -> (1, d)
        y_new = np.array([-next_score])
        model.update(X_new, y_new)

        # Compute metrics
        found_top1 = set(queried_indices) & top1_indices
        retrieval = len(found_top1) / len(top1_indices)
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
            print(f"\n--- Iteration {iteration+n_initial+1} ---")
            print(f"  Top-1% retrieval: {retrieval:.2f} ({len(found_top1)}/{len(top1_indices)})")
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
