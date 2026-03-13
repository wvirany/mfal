"""
A collection of factory functions to generate AnalyticOracle objects for different test functions.
"""

import torch
from botorch.test_functions import Ackley, Branin, Hartmann, Rosenbrock

from molbo.oracle import AnalyticOracle


def gaussian_mixture_1d(noise_std=0.0):
    """Multi-modal 1D test function - mixture of 6 Gaussians."""

    def pdf(x, mean, std):
        return (1 / (std * torch.sqrt(2 * torch.tensor(torch.pi)))) * torch.exp(
            -((x - mean) ** 2) / (2 * std**2)
        )

    def f(X):
        g1 = 0.3 * pdf(X, mean=0.0, std=1.0)
        g2 = 0.25 * pdf(X, mean=2.2, std=0.6)
        g3 = 0.2 * pdf(X, mean=4.2, std=0.7)
        g4 = 0.2 * pdf(X, mean=5.8, std=0.7)
        g5 = 0.5 * pdf(X, mean=8.0, std=1.2)
        g6 = 0.1 * pdf(X, mean=10.1, std=0.6)
        return 10 * (g1 + g2 + g3 + g4 + g5 + g6)

    return AnalyticOracle(
        f=f,
        bounds=torch.tensor([0.0, 10.0]).unsqueeze(-1),
        dim=1,
        noise_std=noise_std,
        optimal_value=1.791,
    )


def ackley(dim=5, noise_std=0.0):
    """Ackley function (arbitrary dim). Negated for maximization."""
    fn = Ackley(dim=dim, negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=dim,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def branin(noise_std=0.0):
    """Branin function (2D). Negated for maximization."""
    fn = Branin(negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=2,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def hartmann(dim=6, noise_std=0.0):
    """Hartmann function (3 or 6D). Negated for maximization."""
    fn = Hartmann(dim=dim, negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=dim,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def rosenbrock(dim=10, noise_std=0.0):
    """Rosenbrock function (arbitrary dim). Negated for maximization."""
    fn = Rosenbrock(dim=dim, negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=dim,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )
