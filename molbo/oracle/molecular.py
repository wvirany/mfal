"""
Named oracle constructors for known molecular datasets.

These thin wrappers exist for Hydra config compatibility — Hydra requires a single
callable _target_, so we can't express oracle_from_dataset() with a nested dataset
object cleanly in YAML. These functions delegate to oracle_from_dataset() internally
and are the config-facing API for datasets.
"""

from molbo.dataset.mcl1 import Mcl1Dataset
from molbo.oracle.factory import oracle_from_dataset


def mcl1_qed(noise_std: float = 0.0, n: int = None):
    dataset = Mcl1Dataset()
    return oracle_from_dataset(dataset, column="qed", noise_std=noise_std, n=n)


def mcl1_vina(noise_std: float = 0.0, n: int = None):
    dataset = Mcl1Dataset()
    return oracle_from_dataset(dataset, column="vina", negate=True, noise_std=noise_std, n=n)


def mcl1_mmgbsa(noise_std: float = 0.0, n: int = None):
    dataset = Mcl1Dataset()
    return oracle_from_dataset(dataset, column="mmgbsa", negate=True, noise_std=noise_std, n=n)
