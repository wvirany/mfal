from abc import ABC, abstractmethod

import wandb


class Logger(ABC):
    """Abstract base class for BO loggers.

    Implement this interface to plug any logging backend into a BOLoop.
    The logger receives a dictionary of metrics at each iteration via `log()`.

    Example:

        class PrintLogger(Logger):
            def log(self, data: dict) -> None:
                print(data)
    """

    @abstractmethod
    def log(self, data: dict) -> None:
        """Log a dictionary of metrics.

        Args:
            data: Metric names mapped to scalar values, e.g. {"best_observed": 0.91, "iteration": 3}.
        """
        ...


class WandBLogger(Logger):
    def __init__(self, project: str, name: str = None, config: dict = None, **kwargs):
        self.run = wandb.init(project=project, name=name, config=config, **kwargs)

    def log(self, data: dict) -> None:
        self.run.log(data)

    def finish(self) -> None:
        self.run.finish()
