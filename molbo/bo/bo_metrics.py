import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from molbo.utils.helpers import get_centroid_indices

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
        smiles: list = None,
        logger=None,
    ):
        self.f_max = f_max
        self.thresholds = thresholds
        self.threshold_labels = threshold_labels
        self.n_top_k = n_top_k
        self.smiles = smiles
        self.logger = logger
        self.history = None

    def initialize(self, history):
        self.history = history

    def update(self, iteration, all_acq_values=None):
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
            "simple_regret": self._compute_simple_regret(y)[-1].item(),
            "cumulative_regret": self._compute_cumulative_regret(y)[-1].item(),
        }

        if self.thresholds is not None:
            for k, threshold in self.thresholds.items():
                rr = self._compute_retrieval_rate(y, threshold, self.n_top_k[k])
                metrics_dict[f"retrieval_rate_{self.threshold_labels[k]}"] = rr[-1].item()

        if all_acq_values is not None:
            metrics_dict["acq_sparsity"] = self._compute_acq_sparsity(all_acq_values)

        # Save to history
        for k, v in metrics_dict.items():
            if k in HISTORY_SKIP:
                continue
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        if self.logger is not None:
            self.logger.log(metrics_dict)

    def _compute_simple_regret(self, y):
        """Compute simple regret at each iteration."""
        return self.f_max - y.cummax(dim=0).values

    def _compute_cumulative_regret(self, y):
        """Compute cumulative regret at each iteration."""
        return torch.cumsum(self.f_max - y, dim=0)

    def _compute_best_observed(self, y):
        """Compute best observed value at each iteration."""
        return torch.cummax(y, dim=0).values

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

    def _compute_retrieval_rate(self, y, threshold, n_top_k):
        """Compute proportion of samples found above given threshold"""
        found = (y >= threshold).cumsum(dim=0)
        return found / n_top_k

    def _compute_acq_sparsity(self, all_acq_values):
        return 1 - (all_acq_values.mean() / all_acq_values.max()).item()

    def compute_metrics(self):
        """
        Compute all post-hoc diversity metrics. This is typically called after a BO run
        by loading `history` from a checkpoint and computing post-hoc; it's expensive to compute
        with WandB during a job.
        """
        if self.smiles is None or self.thresholds is None:
            return

        curves = self.compute_batch_metrics()

        if self.logger is not None:
            n_iters = len(self.history["iteration"])
            y_obs = self.history["y_observed"].squeeze()
            q = len(y_obs) // n_iters
            for i in range(q, len(y_obs) + 1, q):
                log_dict = {"n_observed": i}
                for key, curve in curves.items():
                    log_dict[key] = curve[(i // q) - 1]
                self.logger.log(log_dict)

        return curves

    def compute_batch_metrics(self):
        """
        Outer loop over batches; compute num modes and num_scaffolds curves
        for each threshold in acquisition order.
        """
        y_obs = self.history["y_observed"].reshape(-1)
        observed_indices = self.history["observed_indices"]
        n_iters = len(self.history["iteration"])
        q = len(y_obs) // n_iters
        stride = max(
            1, n_iters // 100
        )  # For reducing computations on runs with many iters (small q, large n)

        mode_curves = {f"num_modes_{self.threshold_labels[k]}": [] for k in self.thresholds}
        scaffold_curves = {f"num_scaffolds_{self.threshold_labels[k]}": [] for k in self.thresholds}
        diversity_curve = []

        for i in range(q, len(y_obs) + 1, q * stride):
            acq_indices = observed_indices[:i]
            acq_smiles = [self.smiles[idx] for idx in acq_indices]
            acq_scores = y_obs[:i]

            # Modes
            num_modes = self._compute_num_modes(acq_smiles, acq_scores)
            for k in self.thresholds:
                mode_curves[f"num_modes_{self.threshold_labels[k]}"].append(num_modes[k])

            # Scaffolds
            for k, threshold in self.thresholds.items():
                above_threshold = [
                    smi for smi, s in zip(acq_smiles, acq_scores) if s.item() >= threshold
                ]
                count, _ = self._compute_num_scaffolds(above_threshold)
                scaffold_curves[f"num_scaffolds_{self.threshold_labels[k]}"].append(count)

            # Batch diversity - mean pairwise Tanimoto distance computed *per batch*
            batch_indices = observed_indices[i - q : i]
            batch_smiles = [self.smiles[idx] for idx in batch_indices]
            diversity_curve.append(self._compute_batch_diversity(batch_smiles))

        return {**mode_curves, **scaffold_curves, "batch_diversity": diversity_curve}

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

    def _compute_num_scaffolds(self, smiles, seen_scaffolds: set = None):
        """
        Number of unique Bemis-Murcko scaffolds in given SMILES list.
        Optionally builds on an existing set of scaffolds.
        """
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
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

        if len(fps) < 2:
            return 0.0

        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

        return 1.0 - np.mean(sims)
