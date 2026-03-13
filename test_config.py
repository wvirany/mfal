import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Print Hydra choices (which config files were selected)
    choices = HydraConfig.get().runtime.choices
    print("Choices:", dict(choices))

    # Construct WandB group name
    mean_name = choices.get("mean_module", "mll") or "mll"
    group = f"{choices['oracle']}_{mean_name}_{choices['acquisition']}_{choices['model']}"
    run_name = f"{group}_seed{cfg.seed}"
    print("WandB group:", group)
    print("WandB run name:", run_name)

    print(choices["oracle"])


if __name__ == "__main__":
    main()
