import argparse
from pathlib import Path

from molbo.utils import WandBResults

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="molbo")
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    results = WandBResults(args.project, experiment=args.experiment)
    results.fetch_runs()
    results.fetch_history()

    results_path = Path(f"results/{args.experiment}.pkl")
    results.save(results_path)
