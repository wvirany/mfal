"""Run single-fidelity active learning baseline."""

import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import wandb
from mfal.data import compute_top_k_indices, get_oracle_function, load_mcl1_data
from mfal.utils.al_loop import run_al_loop
from mfal.utils.embeddings import get_embeddings

warnings.filterwarnings("ignore")


# Get repo root (two levels up from this file)
REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "baselines"

# Set wandb directory (before any wandb imports/init)
WANDB_DIR = REPO_ROOT / "wandb"
WANDB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["WANDB_DIR"] = str(WANDB_DIR)


def initialize_model(model_type: str):
    """Initialize model based on type."""
    if model_type == "blr":
        from mfal.models.bayesian_ridge import BayesianRidgeModel

        return BayesianRidgeModel()
    elif model_type == "random":
        from mfal.models.random_model import RandomModel

        return RandomModel()
    elif model_type == "gp":
        # TODO: Refactor GP
        raise NotImplementedError("GP model not implemented yet")
        from mfal.models.gp import TanimotoGP

        return TanimotoGP()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def create_acquisition(acquisition_type: str, random_seed: int = 42):
    """Create acquisition function based on type."""
    if acquisition_type == "random":
        from mfal.acquisition.random import RandomAcquisition

        return RandomAcquisition(random_state=random_seed)
    elif acquisition_type == "ei":
        from mfal.acquisition.expected_improvement import ExpectedImprovement

        return ExpectedImprovement()
    elif acquisition_type == "ucb":
        from mfal.acquisition.ucb import UpperConfidenceBound

        return UpperConfidenceBound(beta=1.0)
    else:
        raise ValueError(f"Invalid acquisition type: {acquisition_type}")


def main(args):
    """Run single-fidelity active learning baseline."""

    # Start timer
    start_time = time.time()

    # Initialize wandb
    if args.wandb_mode != "disabled":
        run_name = f"single_fidelity_{args.embedding}_{args.score_type}_seed{args.seed}"
        run = wandb.init(
            project="MFAL",
            name=run_name,
            config=args,
            tags=["single-fidelity", args.embedding, args.score_type, "baseline"],
            mode=args.wandb_mode,
        )
    else:
        run = None

    print("\n" + "=" * 70)
    print("Baseline Active Learning Experiment")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Acquisition: {args.acquisition}")
    print(f"Score type: {args.score_type}")
    print(f"Iterations: {args.n_iterations:,}")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

    # Load data
    print("Loading data...")
    df = load_mcl1_data()
    smiles = df["prot_smiles"].tolist()
    print(f"Loaded {len(smiles)} molecules")

    # Get oracle and ground truth
    print(f"Setting up {args.score_type} score oracle...")
    oracle_fn, all_scores = get_oracle_function(df, args.score_type)
    top1_indices, top1_threshold = compute_top_k_indices(all_scores, k_percent=1.0)
    print(f"Top-1% threshold: {top1_threshold:.3f}")

    if run is not None:
        wandb.config.update(
            {
                "n_molecules": len(smiles),
                "top1_threshold": top1_threshold,
                "n_top1_molecules": len(top1_indices),
            }
        )

    # Get embeddings
    print("Getting embeddings...")
    embeddings = get_embeddings(smiles, args.embedding)
    print(f"Embedding shape: {embeddings.shape}")

    # Create model and acquisition function
    print(f"\nInitializing {args.model} model...")
    model = initialize_model(args.model)

    print(f"Initializing {args.acquisition} acquisition function...")
    acq_func = create_acquisition(args.acquisition, random_seed=args.seed)

    # Run AL
    results = run_al_loop(
        embeddings=embeddings,
        oracle_fn=oracle_fn,
        top1_indices=top1_indices,
        model=model,
        acq_func=acq_func,
        n_iterations=args.n_iterations,
        n_initial=args.n_initial,
        score_type=args.score_type,
        random_seed=args.seed,
        verbose=True,
        wandb_run=run,
    )

    # Timing
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    # Log final summary to wandb
    if run is not None:
        wandb.summary.update(
            {
                "final_top1_retrieval": results["top1_retrieval"][-1],
                "final_best_score": results["best_scores"][-1],
                "total_time_seconds": elapsed_time,
                "total_queries": len(results["queried_indices"]),
            }
        )

        # Finish the run
        wandb.finish()

    if args.save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        output_file = (
            RESULTS_DIR / f"{args.model}_{args.acquisition}_{args.score_type}_seed{args.seed}.pkl"
        )
        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default="blr", choices=["blr", "random", "gp"], help="Model type"
    )
    parser.add_argument(
        "--acquisition",
        default="random",
        choices=["random", "ei", "ucb"],
        help="Acquisition function type",
    )
    parser.add_argument(
        "--embedding",
        default="morgan_fp",
        choices=["morgan_fp", "molformer", "gneprop"],
        help="Embedding type",
    )
    parser.add_argument(
        "--score_type", default="docking", choices=["docking", "mmgbsa"], help="Oracle score type"
    )
    parser.add_argument("--n_iterations", type=int, default=3500, help="Number of AL iterations")
    parser.add_argument("--n_initial", type=int, default=100, help="Number of initial points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--wandb_mode",
        default="online",
        choices=["offline", "online", "disabled"],
        help="Wandb mode",
    )
    parser.add_argument("--save_results", action="store_true", help="Save results")
    args = parser.parse_args()

    main(args)
