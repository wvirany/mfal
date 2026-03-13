from molbo.oracle.base import Oracle


class AnalyticOracle(Oracle):

    def __init__(self, f, bounds, dim, noise_std=0.0, optimal_value=None):
        """
        Args:
            f: (Callable) Analytic form of objective function
            bounds: (2, d) tensor matching botorch.optim.optimize_acqf()
            dim: Number of input dimensions
            noise_std: (m) tensor with noise std for each output dim
            optimal_value: Max of function
        """
        super().__init__(noise_std)
        self.f = f
        self.bounds = bounds
        self.dim = dim
        self._optimal_value = optimal_value

    def _evaluate(self, X):
        y = self.f(X)
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # Explicit output dim
        return y

    @property
    def optimal_value(self):
        if self._optimal_value is None:
            raise ValueError("Optimal value not set. Either set during init or compute manually.")
        return self._optimal_value
