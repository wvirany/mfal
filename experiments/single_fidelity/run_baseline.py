"""Run single-fidelity active learning baseline."""

import argparse
import time

from mfal.data import compute_top_k_indices, get_oracle_function, load_mcl1_data
from mfal.utils.al_loop import run_al_loop
from mfal.utils.embeddings import get_embeddings


def main(args):
    """Run single-fidelity active learning baseline."""

    # Start timer
    start_time = time.time()

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
    top1_indices, top1_threshold = compute_top_k_indices(all_scores, k_percent=0.01)
    print(f"Top-1% threshold: {top1_threshold:.3f}")

    # Get embeddings
    print("Getting embeddings...")
    embeddings = get_embeddings(smiles, args.embedding)
    print(f"Embedding shape: {embeddings.shape}")

    # Run AL
    _ = run_al_loop(
        embeddings=embeddings,
        oracle_fn=oracle_fn,
        top1_indices=top1_indices,
        n_iterations=args.n_iterations,
        score_type=args.score_type,
        random_seed=args.seed,
        device=args.device,
        verbose=True,
    )

    # Timing
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


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

    args = parser.parse_args()

    main(args)
