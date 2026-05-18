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
        device: str = "cpu",
    ):
        self.history = history
        self.model = model
        self.acq_func = acq_func
        self.oracle = oracle
        self.acqf_optimizer = acqf_optimizer
        self.candidates = candidates
        self.device = device

        # Initialize model from all data seen so far (handles fresh start and resume)
        X_all = history.X_all
        self.model.initialize(
            X_all.to(device) if torch.is_tensor(X_all) else X_all, history.y_all.to(device)
        )

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
                    self.acq_func, self.candidates, observed_indices=self.history.observed_indices
                )
                new_y = self.oracle(result.observed_indices).to(self.device)

            # Continuous / generative
            else:
                result = self.acqf_optimizer.optimize(self.acq_func)
                new_y = self.oracle(result.new_X).to(self.device)

            # Update model
            self.model.update(result.new_X, new_y)

            # Update history (handles metrics, logging, checkpointing)
            self.history.update(
                result=result,
                new_y=new_y,
                model_loss=self.model.loss().item(),
                iteration=i,
                observed_indices=result.observed_indices,
            )
