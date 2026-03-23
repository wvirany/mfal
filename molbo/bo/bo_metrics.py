import torch


class BOMetrics:
    """
    Metrics for BO loop

    Args:
        f_max: Maximumum of oracle

        top_k_threshold: Threshold to compute num samples above
        n_top_k: Number of samples above top_k_threshold
        --> (Used for computing retrieval rate in fixed datasets)

        logger: WandBLogger instance
    """

    def __init__(
        self, f_max: float, top_k_threshold: float = None, n_top_k: float = None, logger=None
    ):
        self.f_max = f_max
        self.top_k_threshold = top_k_threshold
        self.n_top_k = n_top_k

        self.logger = logger
        self.history = None

    def initialize(self, history):
        self.history = history

    def update(self, iteration):
        y = torch.cat(
            [
                self.history["y_init"],
                self.history["y_observed"],
            ]
        ).squeeze()

        metrics_dict = {
            "iteration": iteration,
            "acq_val": self.history["acq_vals"][-1],
            "time_per_iter": self.history["time_per_iter"][-1],
            "model_loss": self.history["model_loss"][-1],
            "best_observed": self._compute_best_observed(y)[-1].item(),
            "simple_regret": self._compute_simple_regret(y)[-1].item(),
            "cumulative_regret": self._compute_cumulative_regret(y)[-1].item(),
        }

        if self.logger is not None:
            self.logger.log(metrics_dict)

    def compute_metrics(self):
        """Compute post-hoc metrics for finished BO run."""
        y = torch.cat(
            [
                self.history["y_init"],
                self.history["y_observed"],
            ]
        ).squeeze()

        return {
            "simple_regret": self._compute_simple_regret(y),
            "cumulative_regret": self._compute_cumulative_regret(y),
            "best_observed": self._compute_best_observed(y),
            "topk_mean": self._compute_topk_mean(y, k=10),
        }

    def compute_batch_metrics(self):
        raise NotImplementedError("Batch metrics not implemented yet")

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

    def _compute_retrieval_rate(self, y):
        """Compute proportion of samples found above given threshold"""
        if self.top_k_threshold is None:
            return None
        found = (y >= self.top_k_threshold).cumsum(dim=0)
        return found / self.n_top_k
