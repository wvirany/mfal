import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from molbo.bo import BOLoop
from molbo.utils import sample_init


@hydra.main(config_path="config", config_name="config", version_base=None)
def run(cfg: DictConfig):
    # Instantiate objects from config
    oracle = instantiate(cfg.oracle)
    model = instantiate(cfg.model)
    acq_func = instantiate(cfg.acquisition)

    # Sample initial data
    train_X, train_y = sample_init(oracle, cfg.bo.n_init)

    bo = BOLoop(train_X, train_y, model, acq_func, oracle)
    history = bo.run(cfg.bo.n_iters)

    print(f"Best observed: {history['y_observed'].max().item():.4f}")


if __name__ == "__main__":
    run()
