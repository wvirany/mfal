# molbo

A lightweight Bayesian optimization library for molecular design. Supports continuous, fixed-pool, and generative settings.

## Installation

```bash
pip install -e .
```

## Examples

Below are several examples for a quick introduction to using `molbo`. Further documentation can be found [here](#key-ingredients-of-a-boloop).

### Continuous optimization

```python
import torch
from molbo.oracle import AnalyticOracle
from molbo.model.gp import GPModel
from molbo.acquisition.base import EIAcquisition
from molbo.acqf_optimizer.continuous import ContinuousMaximizer
from molbo.bo import BOLoop

def f(X):
    return torch.sin(X) + 0.1 * torch.randn_like(X)

bounds = torch.tensor([[-3.0], [3.0]])

oracle = AnalyticOracle(f=f, dim=1, bounds=bounds, noise_std=0.1)

optimizer = ContinuousMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=5)

model = GPModel()
acq_func = EIAcquisition()

bo_loop = BOLoop(
    train_X=init.train_X,
    train_y=init.train_y,
    model=model,
    acq_func=acq_func,
    oracle=oracle,
    acqf_optimizer=optimizer,
)
bo_loop.run(n_iters=20)
```

### Fixed-pool molecular optimization

```python
import torch
from molbo.oracle.lookup import LookupOracle
from molbo.model.gp import TanimotoGPModel
from molbo.acquisition.base import EIAcquisition
from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.bo import BOLoop
from molbo.utils import smiles_to_morgan_fp

smiles = [
    "C1=C(C2=C(C=C1O)OC(C(C2=O)=O)C3=CC=C(C(=C3)O)O)O",
    "O=S(=O)(N1CCNCCC1)C2=CC=CC=3C2=CC=NC3",
    "C=1C=C2S/C(/N(CC)C2=CC1OC)=C\\C(=O)C",
    "C=1(N=C(C=2C=NC=CC2)C=CN1)NC=3C=C(NC(C4=CC=C(CN5CCN(CC5)C)C=C4)=O)C=CC3C",
    "C1=CC=2C(=CNC2C=C1)C=3C=CN=CC3",
    "N1(C2=C(C(N)=NC=N2)C=N1)C3=CC=CC=C3",
]

X = torch.stack([smiles_to_morgan_fp(smi) for smi in smiles])
y = torch.randn(len(X), dtype=torch.float64).unsqueeze(-1)  # Requires explicit output dim

oracle = LookupOracle(X_data=X, y_data=y)

optimizer = PoolMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=2)

model = TanimotoGPModel()
acq_func = EIAcquisition()

bo_loop = BOLoop(
    train_X=init.train_X,
    train_y=init.train_y,
    model=model,
    acq_func=acq_func,
    oracle=oracle,
    acqf_optimizer=optimizer,
    candidates=X,
    observed_indices=init.observed_indices,
    candidate_smiles=smiles,
)
bo_loop.run(n_iters=4)
```

### Generative molecular optimization

```python
# Coming soon
```

## Key ingredients of a `BOLoop`

A BO loop requires four key ingredients:

- **Oracle**: the objective function being optimized
- **Surrogate model**: a probabilistic model of the objective
- **Acquisition function**: scores candidates using model predictions
- **Acquisition optimizer**: selects the next candidate(s) to evaluate

### Oracles

An oracle wraps the objective function and handles evaluation. THree main types are supported, corresponding to each example setting:

- `AnalyticOracle`: for continuous domains; takes a callable `f` and `bounds`
- `LookupOracle`: for fixed candidate pools; evaluates via hash map lookup
- `MolecularOracle`: for general molecular optimization; takes SMILES strings as input.

    Subclass and implement `_evaluate(smiles) -> (N, 1) tensor`. Example: `QEDOracle`.

### Surrogate models

- `GPModel`: standard GP with RBF kernel; suitable for continuous domains
- `TanimotoGPModel`: GP with Tanimoto kernel; the standard choice for molecular fingerprints

Examples of alternative mean and covariance modules can be found at `molbo.model.modules`

### Acquisition functions

Standard acquisition functions are available in `molbo.acquisition.base`:

- `EIAcquisition`: expected improvement
- `UCBAcquisition`: upper confidence bound
- `TSAcquisition`: Thompson sampling

Batch acquisition functions are available in `molbo.acquisition.monte_carlo`:

- `qEIAcquisition`: MC expected improvement with fantasy conditioning
- `qUCBAcquisition`: MC upper confidence bound

### Acquisition optimizers

The acquisition optimizer determines how candidates are selected given the acquisition function.

**Continuous:**
- `ContinuousMaximizer`: gradient-based optimization via L-BFGS-B
- `ContinuousSampler`: samples proportional to acquisition values over a grid (currently only supported for 1D)

**Fixed-pool:**
- `PoolMaximizer`: exact argmax over candidate set via `optimize_acqf_discrete`
- `PoolSampler`: samples proportional to acquisition values

**Generative**

For the generative setting, the `AcqfOptimizer` base class can be extended to allow for flexible implementations. As an example, we include the `MolGA` class (*coming soon*).


### Notes:

Certain acquisition optimizers only work with specific acquisition functions:

* `ContinuousMaximizer` and `PoolMaximizer` must use a Monte Carlo acquisition function when $q > 1$
* `ContinuousSampler` and `PoolSampler` expect analytic acquisition functions

Moreover, the fixed-pool setting requires a `PoolMaximizer` or `PoolSampler`.

`TSAcquisition` is not compatible with `TanimotoGPModel`.
