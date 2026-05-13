from typing import Optional

import torch
from tqdm import tqdm

from molbo.acqf_optimizer import AcqfOptimizer
from molbo.acquisition import Acquisition
from molbo.bo.history import History
from molbo.model import SurrogateModel
from molbo.oracle import Oracle


class BOLoop:
    """
    BO loop.

    Args:
        history: History object tracking observations, metrics, logging, and checkpointing.
                 Pass a freshly constructed History for a new run, or use History.load()
                 to resume from a checkpoint.
        model: Surrogate model
        acq_func: Acquisition function
        oracle: Oracle
        acqf_optimizer: Acquisition function optimizer
        candidates: (N, d) candidate pool for fixed-pool optimization (optional)
        candidate_smiles: SMILES strings corresponding to candidates (optional)
        device: Device for oracle evaluations
    """

    def __init__(
        self,
        history: History,
        model: SurrogateModel,
        acq_func: Acquisition,
        oracle: Oracle,
        acqf_optimizer: AcqfOptimizer,
        candidates: Optional[torch.Tensor] = None,
        candidate_smiles: Optional[list] = None,
        device: str = "cpu",
    ):
        self.history = history
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle
        self.acqf_optimizer = acqf_optimizer
        self.candidates = candidates
        self.candidate_smiles = candidate_smiles
        self.device = device

        # Initialize model from all data seen so far (handles fresh start and resume)
        self.model.initialize(history.X_all.to(device), history.y_all.to(device))

        # Pool setting: build candidates mask, accounting for already-observed indices on resume
        if candidates is not None:
            self.candidates_mask = torch.ones(len(candidates), dtype=torch.bool)
            if history.observed_indices:
                self.candidates_mask[history.observed_indices] = False

    def run(self, n_iters: int) -> History:
        for i in tqdm(
            range(self.history.start_iteration, n_iters),
            desc="BO",
            unit="iter",
            position=0,
            leave=True,
        ):
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
                if self.candidate_smiles is not None:
                    result.smiles = [self.candidate_smiles[idx] for idx in global_idxs]

            # Continuous / generative
            else:
                result = self.acqf_optimizer.optimize(self.acq_func)
                if result.smiles is not None:
                    new_y = self.oracle(result.smiles).to(self.device)
                else:
                    new_y = self.oracle(result.new_X).to(self.device)

            # Update model
            self.model.update(result.new_X, new_y)

            # Update history (handles metrics, logging, checkpointing)
            self.history.update(
                result=result,
                new_y=new_y,
                model_loss=self.model.loss().item(),
                iteration=i,
                observed_indices=global_idxs if self.candidates is not None else None,
            )
