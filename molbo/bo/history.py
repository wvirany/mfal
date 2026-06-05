from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

from molbo.acqf_optimizer.base import OptimizationResult
from molbo.bo.logger import Logger


def _concat(a: Any, b: Any) -> Any:
    """Append ``b`` onto ``a`` in whatever representation ``a`` uses.

    Tensors are concatenated (with ``b`` detached and moved to CPU); lists
    (e.g. SMILES strings) are extended.
    """
    if torch.is_tensor(a):
        return torch.cat([a, b.detach().cpu()])
    return list(a) + list(b)


class History:
    """Stores and manages the history of a BO run.

    Handles observation storage, metrics computation, logging, and checkpointing.
    Subclass and pass metric callables to add setting-specific metrics.

    Metric callables have signature ``Callable[[History], dict]`` - they receive
    the full history after each update and return a dict of metric names to values.
    External state (e.g. a GFN model) can be captured via closures.

    Candidate inputs (``X_init``, ``X_observed``) are representation-agnostic:
    feature tensors for continuous domains, or lists of general representations
    (e.g. SMILES).

    Args:
        X_init: ``(n_init, d)`` initial training inputs
        y_init: ``(n_init, 1)`` initial training targets
        observed_indices: Pool indices of initial observations (pool setting only)
        metrics: List of callables ``f(history) -> dict`` to compute each iteration
        logger: Logger instance for live logging
        checkpoint_path: If set, save history to this path every ``checkpoint_freq`` iterations
        checkpoint_freq: How often to checkpoint (in iterations)
    """

    def __init__(
        self,
        X_init: Any,
        y_init: torch.Tensor,
        observed_indices: Optional[List[int]] = None,
        metrics: Optional[List[Callable]] = None,
        logger: Optional[Logger] = None,
        checkpoint_path: Optional[Path] = None,
        checkpoint_freq: int = 1,
    ):
        self.X_init = X_init
        self.y_init = y_init
        self.observed_indices = list(observed_indices) if observed_indices else []
        self.metrics = metrics or []
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq

        # Accumulated during run. X_observed mirrors the representation of X_init.
        self.X_observed = [] if isinstance(X_init, list) else torch.tensor([], dtype=X_init.dtype)
        self.y_observed = torch.tensor([], dtype=y_init.dtype)
        self.acq_vals = []
        self.iteration = []
        self.model_loss = []
        self.optimization_time = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        result: OptimizationResult,
        new_y: torch.Tensor,
        model_loss: float,
        iteration: int,
        observed_indices: Optional[List[int]] = None,
    ) -> None:
        """Update history with the results of one BO iteration.

        Args:
            result: Output of ``AcqfOptimizer.optimize()``
            new_y: ``(q, 1)`` oracle evaluations for the proposed candidates
            model_loss: Current model MLL loss
            iteration: Current iteration index
            observed_indices: Global pool indices of evaluated candidates (pool setting only)
        """
        # Append observations
        self.X_observed = _concat(self.X_observed, result.new_X)
        self.y_observed = _concat(self.y_observed, new_y)

        # Scalar bookkeeping
        self.acq_vals.extend(result.acq_val.reshape(-1).tolist())
        self.iteration.append(iteration)
        self.model_loss.append(model_loss)
        self.optimization_time.append(result.optimization_time)

        # Pool-setting indices
        if observed_indices is not None:
            self.observed_indices.extend(observed_indices)

        # Compute and log metrics
        metrics_dict = {"iteration": iteration, "optimization_time": result.optimization_time}
        for metric_fn in self.metrics:
            metrics_dict.update(metric_fn(self))

        if self.logger is not None:
            self.logger.log(metrics_dict)

        # Checkpoint
        if self.checkpoint_path is not None and (iteration + 1) % self.checkpoint_freq == 0:
            self.save(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def start_iteration(self) -> int:
        """Iteration to resume from. Zero for a fresh history, correct on resume."""
        return len(self.iteration)

    @property
    def X_all(self) -> Any:
        """All observed inputs: init + BO iterations."""
        return _concat(self.X_init, self.X_observed)

    @property
    def y_all(self) -> torch.Tensor:
        """All observed targets: init + BO iterations."""
        return torch.cat([self.y_init, self.y_observed])

    @property
    def best_observed(self) -> float:
        """Best target value seen so far."""
        return self.y_all.max().item()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save history and RNG state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "X_init": self.X_init,
                "y_init": self.y_init,
                "X_observed": self.X_observed,
                "y_observed": self.y_observed,
                "observed_indices": self.observed_indices,
                "acq_vals": self.acq_vals,
                "iteration": self.iteration,
                "model_loss": self.model_loss,
                "optimization_time": self.optimization_time,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, **kwargs) -> Optional[History]:
        """Load history from disk and restore RNG state.

        Additional keyword arguments are forwarded to ``__init__`` (e.g. metrics, logger).
        Returns None if the checkpoint does not exist.
        """
        path = Path(path)
        if not path.exists():
            return None

        data = torch.load(path, weights_only=False)

        torch.set_rng_state(data["rng_state"])
        if data.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(data["cuda_rng_state"])

        history = cls(
            X_init=data["X_init"],
            y_init=data["y_init"],
            observed_indices=data["observed_indices"],
            **kwargs,
        )
        history.X_observed = data["X_observed"]
        history.y_observed = data["y_observed"]
        history.acq_vals = data["acq_vals"]
        history.iteration = data["iteration"]
        history.model_loss = data["model_loss"]
        history.optimization_time = data["optimization_time"]

        return history
