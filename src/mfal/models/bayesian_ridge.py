import numpy as np
from sklearn.linear_model import BayesianRidge

from mfal.models.base import SurrogateModel


class BayesianRidgeModel(SurrogateModel):
    """
    Bayesian Ridge Regression model.

    Matches implementation in Andersen et al. (2025)
    """

    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        compute_score: bool = True,
    ):
        """
        Initialize BLR model.

        Args:
            alpha_1: Gamma prior shape parameter on alpha (precision of weights)
            alpha_2: Gamma prior rate parameter on alpha
            lambda_1: Gamma prior shape parameter on lambda (precision of noise)
            lambda_2: Gamma prior rate parameter on lambda
            compute_score: Whether to compute log marginal likelihood
        """
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score

        self.model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score,
        )
        self.is_fitted = False

        # Store training data for incremental updates
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model to training data.
        """
        self.model.fit(X, y)
        self.is_fitted = True

        # Store training data
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation.

        Args:
            X: Test features (n_test, n_features)

        Returns:
            mean: Predicted means (n_test,)
            std: Predicted standard deviations (n_test,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        mean, std = self.model.predict(X, return_std=True)

        return mean, std

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update model with new training data by refitting.

        BLR doesn't support incremental updates, but refitting
        is relatively fast.

        Args:
            X_new: New feature(s) (n_new, n_features)
            y_new: New target(s) (n_new,)
        """
        if not self.is_fitted:
            # First update, juset fit model
            self.fit(X_new, y_new)
        else:
            # Concatenate with existing data
            X_combined = np.vstack([self.X_train, X_new])
            y_combined = np.concatenate([self.y_train, y_new])
            self.fit(X_combined, y_combined)
