import json
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import wandb


def get_run_info(cfg):
    choices = HydraConfig.get().runtime.choices
    experiment_path = choices.get("experiment", "unnamed")
    seed = cfg.seed

    if run_name := cfg.get("run_name"):
        return experiment_path, run_name.replace("/", "_"), seed

    parts = [
        v.split("/")[-1]
        for k, v in choices.items()
        if not k.startswith("hydra") and k not in ("experiment", "seed") and v is not None
    ]
    group = "_".join(parts)

    return experiment_path, group, seed


class WandBLogger:
    """Wrapper for WandB logger."""

    def __init__(
        self,
        project_name: str,
        run_name: str = None,
        group: str = None,
        tags: list = None,
        config: dict = None,
        mode: str = "online",
        silent: bool = True,
    ):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
            mode=mode,
            settings=wandb.Settings(silent=silent),
        )

    @classmethod
    def init_from_cfg(cls, cfg: DictConfig):
        _, run_name, seed = get_run_info(cfg)

        tags = run_name.split("_")
        experiment = cfg.wandb.get("experiment", None)
        if experiment is not None:
            tags.append(experiment)

        return cls(
            project_name=cfg.wandb.project,
            run_name=f"{run_name}_seed{seed}",
            group=run_name,
            tags=tags,
            config=dict(cfg),
            mode=cfg.wandb.mode,
            silent=False,
        )

    def log(self, data: dict):
        self.run.log(data)

    def log_file(self, file_path):
        self.run.save(str(file_path))

    def log_config(self, config: dict):
        self.run.config.update(config)

    def finish(self):
        self.run.finish()


class BOCheckpoint:
    def __init__(self, cfg):
        experiment_path, run_name, seed = get_run_info(cfg)
        self.run_name = run_name
        self.seed = seed
        self.checkpoint_freq = cfg.bo.checkpoint_freq
        self.path = Path("experiments") / experiment_path / run_name / f"seed{seed}.pt"

    def save(self, history):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "history": history,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            self.path,
        )

    def load(self):
        if not self.path.exists():
            return None
        data = torch.load(self.path, weights_only=False)
        torch.set_rng_state(data["rng_state"])
        if data.get("cuda_rng_state", None) is not None:
            torch.cuda.set_rng_state(data["cuda_rng_state"])
        print(f"Resuming {self.run_name} from iteration {len(data['history']['iteration'])}")
        return data["history"]

    def peek_indices(self):
        if not self.path.exists():
            return None
        data = torch.load(self.path, weights_only=False)
        return data["history"].get("indices", None)
