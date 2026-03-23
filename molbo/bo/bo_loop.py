import time

import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from tqdm import tqdm

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
        observed_indices=None,
        sample: bool = False,
        sample_batch_size: int = 1,
        metrics: BOMetrics = None,
    ):
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle

        self.sample = sample
        self.sample_batch_size = sample_batch_size

        self.history = {
            "X_init": train_X.cpu(),
            "y_init": train_y.cpu(),
            "X_observed": torch.tensor([], dtype=torch.float64),
            "y_observed": torch.tensor([], dtype=torch.float64),
            "acq_vals": [],
            "iteration": [],
            "time_per_iter": [],
            "model_loss": [],
        }

        # Initialize candidate set and observed indices mask in fixed-pool setting
        self.candidates = candidates
        assert not (candidates is not None) ^ (
            observed_indices is not None
        ), "candidates and observed_indices must be set together"
        if (candidates is not None) and (observed_indices is not None):
            self.candidates_mask = torch.ones(len(candidates), dtype=torch.bool)
            self.candidates_mask[observed_indices] = False

        # Initialize model
        self.model.initialize(train_X, train_y)

        # Initialize metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.metrics.initialize(self.history)

    def run(self, n_iters):

        for i in tqdm(range(n_iters), desc="BO", unit="iter"):
            iter_start = time.time()

            # Update model and acquisition function
            self.model.fit()
            self.acq_func.update(self.model)

            # Query acquisition function
            new_X, acq_val = self._optimize_acqf_and_get_observation()

            # Evaluate oracle
            new_y = self.oracle(new_X)

            # Update model training data
            self.model.update(new_X, new_y)

            # Track BO loop history
            self.history["time_per_iter"].append(time.time() - iter_start)
            self.history["X_observed"] = torch.cat(
                (self.history["X_observed"], new_X.detach().cpu())
            )
            self.history["y_observed"] = torch.cat(
                (self.history["y_observed"], new_y.detach().cpu())
            )
            self.history["iteration"].append(i)
            self.history["acq_vals"].append(acq_val.item())
            self.history["model_loss"].append(self.model.loss().item())

            if self.metrics is not None:
                self.metrics.update(i)

        self.model.fit()

        return self.history

    def _optimize_acqf_and_get_observation(self):
        if self.candidates is not None:
            filtered_candidates = self.candidates[self.candidates_mask]

            with torch.no_grad():
                if self.sample:
                    acq_values = self.acq_func(filtered_candidates.unsqueeze(1)).squeeze()
                    probs = acq_values / acq_values.sum()
                    local_idx = torch.multinomial(probs, num_samples=1).item()
                    new_X = filtered_candidates[local_idx].unsqueeze(0)
                    acq_val = acq_values[local_idx]
                else:
                    new_X, acq_val = optimize_acqf_discrete(
                        acq_function=self.acq_func.acq_func, q=1, choices=filtered_candidates
                    )
                    local_idx = (filtered_candidates == new_X).all(dim=-1).nonzero()[0].item()

            global_idx = self.candidates_mask.nonzero()[local_idx].item()
            self.candidates_mask[global_idx] = False
        else:
            if self.sample:
                # Sample proportionally - currently 1D only
                X_grid = torch.linspace(
                    self.oracle.bounds[0].item(),
                    self.oracle.bounds[1].item(),
                    1000,
                    dtype=torch.float64,
                ).unsqueeze(-1)
                acq_values = self.acq_func(X_grid.reshape(-1, 1, 1)).squeeze()
                probs = acq_values / acq_values.sum()
                indices = torch.multinomial(probs, num_samples=self.sample_batch_size)
                best_idx = acq_values[indices].argmax()
                new_X = X_grid[indices[best_idx]].unsqueeze(-1)
                acq_val = acq_values[indices[best_idx]]
            else:
                new_X, acq_val = optimize_acqf(
                    acq_function=self.acq_func.acq_func,
                    bounds=self.oracle.bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )
        return new_X, acq_val
