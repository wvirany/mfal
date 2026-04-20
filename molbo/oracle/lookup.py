import torch

from molbo.oracle.base import Oracle


class LookupOracle(Oracle):
    """
    Oracle for fixed candidate pools with pre-computed values implemented as a lookup table.

    Assumes a maximization problem. Currently assumes output dim is 1.
    """

    def __init__(self, X_data, y_data, top_k=0.01, noise_std=0.0):
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

        self.top_k_threshold = torch.quantile(y_data, 1 - top_k).item()
        self.n_top_k = (y_data >= self.top_k_threshold).sum().item()

        self._hash_to_idx = {hash(row.numpy().tobytes()): i for i, row in enumerate(X_data)}

    def _evaluate(self, X):
        """
        Look up values for X in stored dataset.

        Args:
            X: (B, d) tensor

        Returns:
            y: (B, m) tensor

        Shapes:
            X: (B, d)
            X_data: (N, d)
            indices: (B,)
            y_data[indices]: (B, m)
        """
        indices = [self._hash_to_idx[hash(row.cpu().numpy().tobytes())] for row in X]
        return self.y_data[indices]

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

    @property
    def candidates(self):
        return self.X_data

    @property
    def optimal_value(self):
        if self._optimal_value is None:
            raise ValueError("Optimal value not set. Either set during init or compute manually.")
        return self._optimal_value

    def to(self, device):
        self.X_data = self.X_data.to(device)
        self.y_data = self.y_data.to(device)
        return self
