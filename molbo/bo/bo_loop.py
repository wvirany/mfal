import time

import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from molbo.acquisition import Acquisition
from molbo.bo.bo_metrics import BOMetrics
from molbo.models import SurrogateModel
from molbo.oracle import Oracle


class BOLoop:
    """
    BO loop.

    Args:
        model: Surrogate model
        acquisition: Acquisition
        oracle: Oracle
        is_continuous: bool (default True) Whether the input space is continuous or discrete
        candidates: torch.Tensor (default None) B x N x d candidate points for discrete optimization

        metrics: BOMetrics (default None) Handles metrics logging during BO loop
    """

    def __init__(
        self,
        train_X,
        train_y,
        model: SurrogateModel,
        acq_func: Acquisition,
        oracle: Oracle,
        candidates: torch.Tensor = None,
        metrics: BOMetrics = None,
    ):
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle
        self.candidates = candidates
        self.metrics = metrics

        self.history = {
            "X_init": train_X,
            "y_init": train_y,
            "X_observed": torch.tensor([], dtype=torch.float64),
            "y_observed": torch.tensor([], dtype=torch.float64),
            "acq_vals": [],
            "iteration": [],
            "time_per_iter": [],
            "model_loss": [],
        }

        self.model.initialize(train_X, train_y)

        if self.metrics is not None:
            self.metrics.initialize(self.history)

    def run(self, n_iters):

        for i in range(n_iters):
            iter_start = time.time()

            # Update model and acquisition function
            self.model.fit()
            self.acq_func.update(self.model)

            # Query acquisition function
            new_X, acq_val = self.acq_func.get_observation(self.oracle, self.candidates)

            # Evaluate oracle
            new_y = self.oracle(new_X)

            # Update model training data
            self.model.update(new_X, new_y)

            # Track BO loop history
            self.history["time_per_iter"].append(time.time() - iter_start)
            self.history["X_observed"] = torch.cat((self.history["X_observed"], new_X))
            self.history["y_observed"] = torch.cat((self.history["y_observed"], new_y))
            self.history["iteration"].append(i)
            self.history["acq_vals"].append(acq_val.item())
            self.history["model_loss"].append(self.model.loss().item())

            if self.metrics is not None:
                self.metrics.update(i)

        self.model.fit()

        return self.history
