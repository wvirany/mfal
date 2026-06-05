import torch

from molbo.oracle.base import Oracle


class LookupOracle(Oracle):
    """
    Oracle for fixed-pool optimization. Candidates must be pre-featurized by the user
    before passing as X_data. A molbo Featurizer object can be used for this.
    """

    def __init__(self, X_data, y_data, top_k=[0.01], noise_std=0.0):
        """
        Args:
            X_data: (N, d) tensor of input features
            y_data: (N, 1) tensor of outputs
            dim: Number of input dimensions
            noise_std: (1) tensor with noise std for each output dim
        """
        super().__init__(noise_std)

        self.X_data = X_data
        self.y_data = y_data

        self.dim = X_data.shape[-1]
        self._optimal_value = y_data.max().item()

        self.thresholds = {k: torch.quantile(y_data, 1 - k).item() for k in top_k}
        self.threshold_labels = {k: f"top{k * 100:g}pct" for k in top_k}
        self.n_top_k = {k: (y_data >= self.thresholds[k]).sum().item() for k in top_k}

    def _evaluate(self, indices):
        return self.y_data[indices]

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

    @property
    def candidates(self):
        return self.X_data

    @property
    def optimal_value(self):
        return self._optimal_value

    def to(self, device):
        self.X_data = self.X_data.to(device)
        self.y_data = self.y_data.to(device)
        return self
