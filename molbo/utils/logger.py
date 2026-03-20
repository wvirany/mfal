import json
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import wandb


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
        choices = HydraConfig.get().runtime.choices

        # Build tags
        tags = [v for k, v in choices.items() if not k.startswith("hydra") and v is not None]

        group = "_".join(tags)
        run_name = f"{group}_seed{cfg.seed}"

        experiment = cfg.wandb.get("experiment", None)
        if experiment is not None:
            tags.append(experiment)

        return cls(
            project_name=cfg.wandb.project,
            run_name=run_name,
            group=group,
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


class WandBResults:
    def __init__(self, project: str, experiment: str = None):
        self.project = project
        self.experiment = experiment
        self.api = wandb.Api()
        self.runs = None
        self.summary_df = None
        self.history_df = None

    def fetch_runs(self):
        filters = {}
        if self.experiment is not None:
            filters = {"tags": {"$in": [self.experiment]}}

        self.runs = list(self.api.runs(self.project, filters=filters))

        rows = []
        for run in self.runs:
            row = {"run_id": run.id}
            for tag in run.tags:
                row[tag] = True
            summary = json.loads(run.summary._json_dict)
            row.update({k: v for k, v in summary.items() if not k.startswith("_")})
            rows.append(row)

        self.summary_df = pd.DataFrame(rows)

    def fetch_history(self):
        assert self.runs is not None, "Call fetch_runs() first"

        dfs = []
        for run in self.runs:
            df = run.history()
            df["run_id"] = run.id
            for tag in run.tags:
                df[tag] = True
            dfs.append(df)

        self.history_df = pd.concat(dfs, ignore_index=True)
        self.history_df = self.history_df.loc[:, ~self.history_df.columns.str.startswith("_")]
        return self

    def save(self, path: str):
        assert self.summary_df is not None, "No data to save - call fetch_runs() first"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"summary_df": self.summary_df, "history_df": self.history_df}, path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str) -> dict:
        return pd.read_pickle(path)
