import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from molbo.bo import BOLoop, BOMetrics
from molbo.utils import sample_init
from molbo.utils.logger import WandBLogger


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    # Instantiate components
    oracle = instantiate(cfg.oracle)
    acq_func = instantiate(cfg.acquisition)
    model = instantiate(cfg.model)

    # Resolve WandB parameters
    choices = HydraConfig.get().runtime.choices
    c = choices.get("model/mean_module")
    mean_name = c if c is not None else "mll"
    group = f"{choices['oracle']}_{mean_name}_{choices['acquisition']}_{choices['model']}"
    run_name = f"{group}_seed{cfg.seed}"

    tags = [
        choices["oracle"],
        choices["model"],
        mean_name,
        choices["acquisition"],
    ]

    logger = WandBLogger(
        project_name=cfg.wandb.project,
        run_name=run_name,
        group_name=group,
        tags=tags,
        mode=cfg.wandb.mode,
        run_config=dict(cfg),
    )

    # Init data
    train_X, train_y = sample_init(oracle, n_init=cfg.bo.n_init)

    # Run
    metrics = BOMetrics(f_max=oracle.optimal_value, logger=logger)
    bo = BOLoop(train_X, train_y, model, acq_func, oracle, metrics=metrics)
    history = bo.run(n_iters=cfg.bo.n_iters)

    logger.finish()

    print(history["y_observed"].max())


if __name__ == "__main__":
    main()
