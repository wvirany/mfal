import torch

from molbo.data import load_mcl1_data
from molbo.oracle import LookupOracle
from molbo.utils import smiles_to_morgan_fp


def mcl1_qed(noise_std=0.0):
    df = load_mcl1_data()
    smiles_list = df["prot_smiles"].to_list()
    X = torch.vstack([smiles_to_morgan_fp(s) for s in smiles_list])
    y = torch.tensor(df["qed_score"].values, dtype=torch.float64).unsqueeze(-1)
    return LookupOracle(X_data=X, y_data=y, noise_std=noise_std)


def mcl1_vina(noise_std=0.0):
    df = load_mcl1_data()
    smiles_list = df["prot_smiles"].to_list()
    X = torch.vstack([smiles_to_morgan_fp(s) for s in smiles_list])
    y = -torch.tensor(df["vina_score"].values, dtype=torch.float64).unsqueeze(-1)
    return LookupOracle(X_data=X, y_data=y, noise_std=noise_std)


def mcl1_mmgbsa(noise_std=0.0):
    df = load_mcl1_data()
    smiles_list = df["prot_smiles"].to_list()
    X = torch.vstack([smiles_to_morgan_fp(s) for s in smiles_list])
    y = -torch.tensor(df["mmgbsa_score"].values, dtype=torch.float64).unsqueeze(-1)
    return LookupOracle(X_data=X, y_data=y, noise_std=noise_std)
