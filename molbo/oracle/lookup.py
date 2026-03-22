from molbo.oracle.base import Oracle


class LookupOracle(Oracle):
    """
    Oracle for fixed candidate pools with pre-computed values implemented as a lookup table.

    Assumes a maximization problem.
    """

    def __init__(self, X_data, y_data, noise_std=0.0):
        """
        Args:
            X_data: (N, d) tensor of input features
            y_data: (N, m) tensor of outputs
            dim: Number of input dimensions
            noise_std: (m) tensor with noise std for each output dim
        """
        super().__init__(noise_std)

        self.X_data = X_data
        self.y_data = y_data

        self.dim = X_data.shape[-1]
        self._optimal_value = y_data.max().item()

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
            matches = (X_data.unsqueeze(0) == X.reshape(-1, 1, self.dim)).all(dim=-1), dim is along d; (B,)
            indices = matches.int().argmax(dim=1), dim is along N; (B,)
            y_data[indices]
        """
        matches = (self.X_data.unsqueeze(0) == X.reshape(-1, 1, self.dim)).all(dim=-1)
        indices = matches.int().argmax(dim=-1)

        # Check if any points were not found
        if not matches.any(dim=-1).all():
            raise ValueError("Some points not found in dataset")

        return self.y_data[indices]

    @property
    def candidates(self):
        return self.X_data

    @property
    def optimal_value(self):
        if self._optimal_value is None:
            raise ValueError("Optimal value not set. Either set during init or compute manually.")
        return self._optimal_value
