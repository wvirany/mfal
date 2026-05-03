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
        indices=None,
        candidate_smiles=None,
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
            # Save indices and candidates SMILES in cases where they were not originally saved
            # This is used for computing metrics on old runs
            if "indices" not in self.history and indices is not None:
                self.history["indices"] = indices
            if "candidate_smiles" not in self.history and candidate_smiles is not None:
                self.history["candidate_smiles"] = candidate_smiles
        else:
            self.history = {
                "X_init": train_X.cpu(),
                "y_init": train_y.cpu(),
                "X_observed": torch.tensor([], dtype=torch.float64),
                "y_observed": torch.tensor([], dtype=torch.float64),
                "observed_indices": observed_indices,
                "indices": indices,
                "candidate_smiles": candidate_smiles,
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

        for i in tqdm(
            range(self.start_iteration, n_iters), desc="BO", unit="iter", position=0, leave=True
        ):
            iter_start = time.time()

            # Update model and acquisition function
            self.model.fit()
            self.acq_func.update(self.model)

            # Pool-based
            if self.candidates is not None:
                result = self.acqf_optimizer.optimize(
                    self.acq_func, self.candidates[self.candidates_mask]
                )
                hash_idxs = [hash(x.cpu().numpy().tobytes()) for x in result.new_X]
                global_idxs = [self.oracle._hash_to_idx[h] for h in hash_idxs]
                new_y = self.oracle[global_idxs][1]
                if new_y.dim() == 1:
                    new_y = new_y.unsqueeze(-1)
                self.candidates_mask[global_idxs] = False
            # Continuous / generative
            else:
                result = self.acqf_optimizer.optimize(self.acq_func)
                if result.smiles is not None:
                    new_y = self.oracle(result.smiles)
                else:
                    new_y = self.oracle(result.new_X).to(self.device)

            # Update dataset
            self.model.update(result.new_X, new_y)

            # Track BO loop history
            self.history["time_per_iter"].append(time.time() - iter_start)
            self.history["X_observed"] = torch.cat(
                (self.history["X_observed"], result.new_X.detach().cpu())
            )
            self.history["y_observed"] = torch.cat(
                (self.history["y_observed"], new_y.detach().cpu())
            )
            if self.candidates is not None:
                self.history["observed_indices"].extend(global_idxs)
            self.history["iteration"].append(i)
            self.history["acq_vals"].extend(result.acq_val.reshape(-1).tolist())
            self.history["model_loss"].append(self.model.loss().item())

            # Save checkpoint
            if (self.checkpoint is not None) and ((i + 1) % self.checkpoint.checkpoint_freq == 0):
                self.checkpoint.save(self.history)

            # Update metrics
            if self.metrics is not None:
                self.metrics.update(i, extra_metrics=result.metrics)

        if self.candidates is not None and self.metrics is not None:
            self.history["batch_metrics"] = self.metrics.compute_batch_metrics()

        if self.checkpoint is not None:
            self.checkpoint.save(self.history)

        return self.history
