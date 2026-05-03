import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from molbo.bo import BOLoop, BOMetrics
from molbo.oracle.factory import oracle_from_dataset
from molbo.utils.logger import BOCheckpoint, WandBLogger

OmegaConf.register_new_resolver("div", lambda a, b: int(a) // int(b))


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(cfg.run_name)

    # Create checkpoint class for saving / loading
    checkpoint = BOCheckpoint(cfg) if cfg.bo.checkpoint else None

    # Instantiate dataset and oracle
    indices = None
    candidate_smiles = None
    dataset = instantiate(cfg.dataset) if cfg.get("dataset") else None
    if dataset is not None:
        n = cfg.get("n")
        indices = checkpoint.peek_indices() if checkpoint is not None else None
        if indices is None:
            indices = torch.randperm(len(dataset.candidates))[:n] if n is not None else None
        oracle = oracle_from_dataset(
            dataset,
            column=cfg.oracle.column,
            negate=cfg.oracle.get("negate", False),
            noise_std=cfg.oracle.get("noise_std", 0.0),
            indices=indices,
        ).to(device)
        candidates = dataset.candidates[indices] if indices is not None else dataset.candidates
        candidates = candidates.to(device)

        candidate_smiles = dataset.candidate_smiles
        if indices is not None:
            candidate_smiles = [candidate_smiles[i] for i in indices]
    else:
        oracle = instantiate(cfg.oracle).to(device)
        candidates = None

    # Instantiate other components
    acqf_optimizer = instantiate(cfg.acqf_optimizer)
    acqf_optimizer.bounds = getattr(oracle, "bounds", None)
    acq_func = instantiate(cfg.acquisition)
    acq_func.candidates = candidates
    model = instantiate(cfg.model)

    logger = WandBLogger.init_from_cfg(cfg)

    # Init data
    init = acqf_optimizer.sample_init(oracle, n_init=cfg.bo.n_init)
    init.train_X.to(device)
    init.train_y.to(device)

    print("Initial dataset size:", len(init.train_X))

    # Class for computing metrics; top_k_threshold and n_top_k provided by LookupOracle
    metrics = BOMetrics(
        f_max=oracle.optimal_value,
        thresholds=getattr(oracle, "thresholds", None),
        threshold_labels=getattr(oracle, "threshold_labels", None),
        n_top_k=getattr(oracle, "n_top_k", None),
        smiles=candidate_smiles if dataset is not None else None,
        logger=logger,
    )

    # Run
    bo = BOLoop(
        init.train_X,
        init.train_y,
        model,
        acq_func,
        oracle,
        acqf_optimizer,
        candidates=candidates,
        observed_indices=init.observed_indices,
        indices=indices,
        candidate_smiles=candidate_smiles,
        metrics=metrics,
        checkpoint=checkpoint,
        device=device,
    )

    history = bo.run(n_iters=cfg.bo.n_iters)

    logger.finish()

    print("Final dataset size:", len(history["y_init"]) + len(history["y_observed"]))
    print(f"Best observed value: {history['y_observed'].max().item():.4f}")


if __name__ == "__main__":
    main()
