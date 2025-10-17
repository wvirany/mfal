"""Run single-fidelity active learning baseline."""

import argparse
import pickle
import time
import warnings
from pathlib import Path

import wandb

from mfal.data import compute_top_k_indices, get_oracle_function, load_mcl1_data
from mfal.utils.al_loop import run_al_loop
from mfal.utils.embeddings import get_embeddings

warnings.filterwarnings("ignore")


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
    print("Single-Fidelity Active Learning Baseline")
    print("=" * 70)
    print(f"Embedding: {args.embedding}")
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

    # Run AL
    results = run_al_loop(
        embeddings=embeddings,
        oracle_fn=oracle_fn,
        top1_indices=top1_indices,
        n_iterations=args.n_iterations,
        score_type=args.score_type,
        random_seed=args.seed,
        device=args.device,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.embedding}_{args.score_type}_seed{args.seed}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="results/single_fidelity", help="Output directory")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use"
    )
    parser.add_argument(
        "--wandb_mode",
        default="online",
        choices=["offline", "online", "disabled"],
        help="Wandb mode",
    )

    args = parser.parse_args()

    main(args)
