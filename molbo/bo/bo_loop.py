import time

import torch
from tqdm import tqdm

from molbo.acqf_optimizer import AcqfOptimizer
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
        acqf_optimizer: AcqfOptimizer,
        candidates: torch.Tensor = None,
        observed_indices=None,
        metrics: BOMetrics = None,
        checkpoint=None,
        device="cpu",
    ):
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle
        self.acqf_optimizer = acqf_optimizer

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
            self.candidates_mask = torch.ones(
                len(candidates), dtype=torch.bool, device=candidates.device
            )
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

            # Pool-based
            if self.candidates is not None:
                new_X, acq_val = self.acqf_optimizer.optimize(
                    self.acq_func, self.candidates[self.candidates_mask]
                )
                hash_idxs = [hash(x.cpu().numpy().tobytes()) for x in new_X]
                global_idxs = [self.oracle._hash_to_idx[h] for h in hash_idxs]
                new_y = self.oracle[global_idxs][1]
                if new_y.dim() == 1:
                    new_y = new_y.unsqueeze(-1)
                self.candidates_mask[global_idxs] = False
            # Continuous / generative
            else:
                new_X, acq_val = self.acqf_optimizer.optimize(self.acq_func)
                new_y = self.oracle(new_X)

            # Update dataset
            self.model.update(new_X, new_y)

            # Track BO loop history
            self.history["time_per_iter"].append(time.time() - iter_start)
            self.history["X_observed"] = torch.cat(
                (self.history["X_observed"], new_X.detach().cpu())
            )
            self.history["y_observed"] = torch.cat(
                (self.history["y_observed"], new_y.detach().cpu())
            )
            if self.candidates is not None:
                self.history["observed_indices"].extend(global_idxs)
            self.history["iteration"].append(i)
            self.history["acq_vals"].extend(acq_val.reshape(-1).tolist())
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
