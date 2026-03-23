import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from molbo.bo import BOLoop, BOMetrics
from molbo.utils import sample_init
from molbo.utils.logger import BOCheckpoint, WandBLogger


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate components
    oracle = instantiate(cfg.oracle).to(device)
    acq_func = instantiate(cfg.acquisition)
    model = instantiate(cfg.model)

    logger = WandBLogger.init_from_cfg(cfg)

    # Init data
    train_X, train_y, observed_indices = sample_init(oracle, n_init=cfg.bo.n_init)

    # Class for computing metrics; top_k_threshold and n_top_k used by LookupOracle
    metrics = BOMetrics(
        f_max=oracle.optimal_value,
        logger=logger,
        top_k_threshold=getattr(oracle, "top_k_threshold", None),
        n_top_k=getattr(oracle, "n_top_k", None),
    )
    candidates = getattr(oracle, "candidates", None)

    # Create checkpoint class for saving / loading
    checkpoint = BOCheckpoint(cfg) if cfg.bo.checkpoint else None

    # Run
    bo = BOLoop(
        train_X,
        train_y,
        model,
        acq_func,
        oracle,
        candidates=candidates,
        observed_indices=observed_indices,
        metrics=metrics,
        checkpoint=checkpoint,
    )
    history = bo.run(n_iters=cfg.bo.n_iters)

    logger.finish()

    print(history["y_observed"].max())


if __name__ == "__main__":
    main()
