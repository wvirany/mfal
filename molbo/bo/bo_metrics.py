import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from molbo.utils.helpers import get_centroid_indices, smiles_to_morgan_fp

# Values to log to WandB but skip in checkpointing
HISTORY_SKIP = {"iteration", "acq_val", "time_per_iter", "model_loss"}


class BOMetrics:
    """
    Metrics for BO loop

    Args:
        f_max: Maximum of oracle
        thresholds: Dict mapping top-k fraction to score threshold, e.g. {0.01: 0.9, 0.10: 0.7}
        n_top_k: Dict mapping top-k fraction to count, e.g. {0.01: 500, 0.10: 5000}
        smiles: SMILES list for candidate pool, indexed consistently with candidates
        logger: WandBLogger instance
    """

    def __init__(
        self,
        f_max: float,
        thresholds: dict = None,
        threshold_labels: dict = None,
        n_top_k: dict = None,
        logger=None,
    ):
        self.f_max = f_max
        self.thresholds = thresholds
        self.threshold_labels = threshold_labels
        self.n_top_k = n_top_k
        self.logger = logger
        self.history = None

        self.seen_scaffolds = {}
        if self.thresholds is not None:
            self.seen_scaffolds = {k: set() for k in self.thresholds}
        self.seen_scaffolds["all"] = set()

    def initialize(self, history):
        self.history = history
        if "seen_scaffolds" in history:
            self.seen_scaffolds = history["seen_scaffolds"]

    def print_summary(self):
        y_init = self.history["y_init"].reshape(-1)
        y_obs = self.history["y_observed"].reshape(-1)
        y = torch.cat([y_init, y_obs])

        print("\n=== BO Summary ===")
        print(f"Total observations: {len(y_obs)}")
        print(f"Best observed: {self._compute_best_observed(y)[-1].item():.4f}")
        print(f"Top-10 mean: {self._compute_topk_mean(y, k=10)[-1].item():.4f}")

        if self.f_max is not None:
            print(f"Simple regret: {self._compute_simple_regret(y)[-1].item():.4f}")

        if self.thresholds is not None:
            print("\nRetrieval rate:")
            for k in reversed(sorted(self.thresholds)):
                print(
                    f"  {self.threshold_labels[k]}: {self.history[f'retrieval_rate_{self.threshold_labels[k]}'][-1]:.4f}"
                )

            print("\nNum scaffolds:")
            print(f"  total: {len(self.seen_scaffolds['all'])}")
            for k in reversed(sorted(self.thresholds)):
                print(f"  {self.threshold_labels[k]}: {len(self.seen_scaffolds[k])}")

    def update(self, iteration, new_y, new_smiles=None, extra_metrics=None):
        y_init = self.history["y_init"].reshape(-1)
        y_obs = self.history["y_observed"].reshape(-1)
        y = torch.cat([y_init, y_obs])

        metrics_dict = {
            "iteration": iteration,
            "n_observed": len(y_obs),
            "acq_val": self.history["acq_vals"][-1],
            "time_per_iter": self.history["time_per_iter"][-1],
            "model_loss": self.history["model_loss"][-1],
            "best_observed": self._compute_best_observed(y)[-1].item(),
            "top10_mean": self._compute_topk_mean(y, k=10)[-1].item(),
        }

        if self.f_max is not None:
            metrics_dict["simple_regret"] = self._compute_simple_regret(y)[-1].item()
            metrics_dict["cumulative_regret"] = self._compute_cumulative_regret(y)[-1].item()

        if self.thresholds is not None:
            for k, threshold in self.thresholds.items():
                rr = self._compute_retrieval_rate(y, threshold, self.n_top_k[k])
                metrics_dict[f"retrieval_rate_{self.threshold_labels[k]}"] = rr[-1].item()

        if new_smiles is not None:
            _, self.seen_scaffolds["all"] = self._compute_num_scaffolds(
                new_smiles, self.seen_scaffolds["all"]
            )
            metrics_dict["num_scaffolds"] = len(self.seen_scaffolds["all"])
            if self.thresholds is not None:
                for k, threshold in self.thresholds.items():
                    above_threshold = [
                        smi
                        for smi, score in zip(new_smiles, new_y.reshape(-1))
                        if score.item() >= threshold
                    ]
                    _, self.seen_scaffolds[k] = self._compute_num_scaffolds(
                        above_threshold, self.seen_scaffolds[k]
                    )
                    metrics_dict[f"num_scaffolds_{self.threshold_labels[k]}"] = len(
                        self.seen_scaffolds[k]
                    )

        # Batch metrics
        if len(new_y.reshape(-1)) > 1:
            batch_y = new_y.reshape(-1)
            metrics_dict["batch_mean"] = batch_y.mean().item()
            if new_smiles is not None:
                metrics_dict["batch_diversity"] = self._compute_batch_diversity(new_smiles)

        if extra_metrics is not None:
            metrics_dict.update(extra_metrics)

        # Save to history
        for k, v in metrics_dict.items():
            if k in HISTORY_SKIP:
                continue
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)
        self.history["seen_scaffolds"] = self.seen_scaffolds

        if self.logger is not None:
            self.logger.log(metrics_dict)

    def _compute_simple_regret(self, y):
        """Compute simple regret at each iteration."""
        return self.f_max - y.cummax(dim=0).values

    def _compute_cumulative_regret(self, y):
        """Compute cumulative regret at each iteration."""
        return torch.cumsum(self.f_max - y, dim=0)

    def _compute_topk_mean(self, y, k=10):
        """Compute mean of top-k of observed values at each iteration."""
        topk_means = []
        for i in range(1, len(y) + 1):
            if i >= k:
                topk_values, _ = torch.topk(y[:i], k)
                topk_means.append(topk_values.mean().item())
            else:
                topk_means.append(y[:i].mean().item())
        return torch.tensor(topk_means)

    def _compute_best_observed(self, y):
        """Compute best observed value at each iteration."""
        return torch.cummax(y, dim=0).values

    def _compute_retrieval_rate(self, y, threshold, n_top_k):
        """Compute proportion of samples found above given threshold"""
        found = (y >= threshold).cumsum(dim=0)
        return found / n_top_k

    def _compute_num_scaffolds(self, smiles, seen_scaffolds=None):
        if seen_scaffolds is None:
            seen_scaffolds = set()
        for smi in smiles:
            try:
                scaffold = MurckoScaffoldSmiles(smi)
                seen_scaffolds.add(scaffold)
            except Exception:
                continue
        return len(seen_scaffolds), seen_scaffolds

    def _compute_batch_diversity(self, smiles):
        """
        Compute batch diversity in terms of pairwise Tanimoto similarity.
        """
        fps = []
        for smi in smiles:
            try:
                fps.append(smiles_to_morgan_fp(smi, as_tensor=False))
            except ValueError:
                continue

        if len(fps) < 2:
            return 0.0

        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

        return 1.0 - np.mean(sims)

    def _compute_num_modes(self, smiles, scores):
        """
        Greedy Tanimoto clustering (threshold=0.7) in descending score order.
        Returns count of mode centroids above each score threshold.
        """
        centroid_indices = get_centroid_indices(smiles, scores)
        centroid_scores = scores[centroid_indices]

        return {
            k: (centroid_scores >= threshold).sum().item()
            for k, threshold in self.thresholds.items()
        }
