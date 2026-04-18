import time

import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from tqdm import tqdm

from molbo.acquisition import Acquisition
from molbo.bo.bo_metrics import BOMetrics
from molbo.model import SurrogateModel
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
        checkpoint=None,
        device="cpu",
    ):
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle

        self.sample = sample
        self.sample_batch_size = sample_batch_size

        # Initialize from checkpoint if it exists
        self.checkpoint = checkpoint
        loaded_history = self.checkpoint.load() if self.checkpoint is not None else None

        if loaded_history is not None:
            self.history = loaded_history
            self.start_iteration = len(self.history["iteration"])
            train_X = torch.cat([self.history["X_init"], self.history["X_observed"]]).to(device)
            train_y = torch.cat([self.history["y_init"], self.history["y_observed"]]).to(device)
            observed_indices = self.history["observed_indices"]
        else:
            self.history = {
                "X_init": train_X.cpu(),
                "y_init": train_y.cpu(),
                "X_observed": torch.tensor([], dtype=torch.float64),
                "y_observed": torch.tensor([], dtype=torch.float64),
                "observed_indices": observed_indices,
                "acq_vals": [],
                "iteration": [],
                "time_per_iter": [],
                "model_loss": [],
            }
            self.start_iteration = 0

        # Initialize candidate set and observed indices mask in fixed-pool setting
        self.candidates = candidates
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

        for i in tqdm(range(self.start_iteration, n_iters), desc="BO", unit="iter"):
            iter_start = time.time()

            # Update model and acquisition function
            self.model.fit()
            self.acq_func.update(self.model)

            # Query acquisition function
            new_X, acq_val, idx = self._optimize_acqf_and_get_observation()

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
            if idx is not None:
                self.history["observed_indices"].append(idx)
            self.history["iteration"].append(i)
            self.history["acq_vals"].append(acq_val.item())
            self.history["model_loss"].append(self.model.loss().item())

            # Save checkpoint
            if (self.checkpoint is not None) and ((i + 1) % self.checkpoint.checkpoint_freq == 0):
                self.checkpoint.save(self.history)

            # Update metrics
            if self.metrics is not None:
                self.metrics.update(i)

        if self.checkpoint is not None:
            self.checkpoint.save(self.history)

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
                        acq_function=self.acq_func.acq_func,
                        q=1,
                        choices=filtered_candidates,
                        max_batch_size=1024,
                    )
                    local_idx = (filtered_candidates == new_X).all(dim=-1).nonzero()[0].item()

            idx = self.candidates_mask.nonzero()[local_idx].item()
            self.candidates_mask[idx] = False
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
            idx = None

        return new_X, acq_val, idx
