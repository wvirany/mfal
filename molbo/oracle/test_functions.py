"""
A collection of factory functions to generate AnalyticOracle objects for different test functions.
"""

import torch
from botorch.test_functions import (
    Ackley,
    Branin,
    EggHolder,
    Hartmann,
    Michalewicz,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
)

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


def eggholder(noise_std=0.0):
    """Eggholder function (2D). Negated for maximization."""
    fn = EggHolder(negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=2,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def goldstein_price(noise_std=0.0):
    """Goldstein-Price function (2D). Negated for maximization."""
    bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])

    def f(X):
        x1, x2 = X[..., 0], X[..., 1]
        a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        b = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return -(a * b)  # negate for maximization

    return AnalyticOracle(
        f=f,
        bounds=bounds,
        dim=2,
        noise_std=noise_std,
        optimal_value=-3.0,
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


def michalewicz(dim=10, noise_std=0.0):
    """Michalewicz function (arbitrary dim). Negated for maximization."""
    fn = Michalewicz(dim=dim, negate=True)
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


def shekel(noise_std=0.0):
    """Shekel function (4D). Negated for maximization."""
    fn = Shekel(negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=4,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def six_hump_camel(noise_std=0.0):
    """Six-hump camel function (2D). Negated for maximization."""
    fn = SixHumpCamel(negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=2,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )


def styblinski_tang(dim=10, noise_std=0.0):
    """Styblinski-Tang function (arbitrary dim). Negated for maximization."""
    fn = StyblinskiTang(dim=dim, negate=True)
    return AnalyticOracle(
        f=lambda X: fn(X),
        bounds=fn.bounds,
        dim=dim,
        noise_std=noise_std,
        optimal_value=fn.optimal_value,
    )
