# molbo

A lightweight Bayesian optimization library for molecular design.

## Installation

Install the latest version:

```bash
pip install git+https://github.com/wvirany/molbo
```

## Usage

molbo is easy to use! It just requires the following key ingredients:

- **Oracle**: the objective function being optimized
- **Surrogate model**: a probabilistic model of the objective
- **Acquisition function**: scores candidates using model predictions
- **Acquisition optimizer**: selects the next candidate(s) to evaluate
- **History**: handles bookkeeping, metric computation, logging, checkpointing


molbo is built on top of BoTorch and supports three optimization settings, each defined by a specific oracle and acquisition optimizer pairing:

**Continuous setting:** in the continuous setting, `AnalyticOracle` wraps a differentiable function and `ContinuousMaximizer` optimizes the acquisition function via gradient-based optimization over a bounded domain.

**Fixed pool setting:** In the fixed-pool setting, `LookupOracle` holds a fixed pre-featurized candidate set and `PoolMaximizer` or `PoolSampler` selects from it.

**General setting:** In the general setting, an oracle can be defined to evaluate arbitrary objects proposed by a custom `AcqfOptimizer` subclass - e.g., a generative model. An example of this is defining a `MolecularOracle` class which evaluates SMILES strings.

Across all three settings, the surrogate model and acquisition function are wrappers built on BoTorch objects which operate on tensors. It is up to the user to define how objects are featurized into tensors.

The following demonstrates several examples for a quick introduction to using `molbo`. Further documentation can be found below.

### Continuous BO

```python
import torch
from molbo.oracle import AnalyticOracle
from molbo.model.gp import GPModel
from molbo.acquisition.base import EIAcquisition
from molbo.acqf_optimizer.continuous import ContinuousMaximizer
from molbo.bo import BOLoop, History

def f(X):
    return torch.sin(X) + 0.1 * torch.randn_like(X)

bounds = torch.tensor([[-3.0], [3.0]])
oracle = AnalyticOracle(f=f, dim=1, bounds=bounds, noise_std=0.1)

optimizer = ContinuousMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=5)

model = GPModel()
acq_func = EIAcquisition()
history = History(X_init=init.train_X, y_init=init.train_y)

bo_loop = BOLoop(
    history=history,
    model=model,
    acq_func=acq_func,
    oracle=oracle,
    acqf_optimizer=optimizer,
)
bo_loop.run(n_iters=20)
```

### Fixed-pool BO

```python
import torch
from molbo.oracle.lookup import LookupOracle
from molbo.model.gp import TanimotoGPModel
from molbo.model.featurizer import MorganFingerprintFeaturizer
from molbo.acquisition.base import EIAcquisition
from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.bo import BOLoop, History

smiles = [
    "C1=C(C2=C(C=C1O)OC(C(C2=O)=O)C3=CC=C(C(=C3)O)O)O",
    "O=S(=O)(N1CCNCCC1)C2=CC=CC=3C2=CC=NC3",
    "C=1C=C2S/C(/N(CC)C2=CC1OC)=C\\C(=O)C",
    "C=1(N=C(C=2C=NC=CC2)C=CN1)NC=3C=C(NC(C4=CC=C(CN5CCN(CC5)C)C=C4)=O)C=CC3C",
    "C1=CC=2C(=CNC2C=C1)C=3C=CN=CC3",
    "N1(C2=C(C(N)=NC=N2)C=N1)C3=CC=CC=C3",
]

# Featurization is the caller's responsibility in the fixed-pool setting
featurizer = MorganFingerprintFeaturizer()
X = featurizer(smiles)
y = torch.randn(len(X), dtype=torch.float64).unsqueeze(-1)

oracle = LookupOracle(X_data=X, y_data=y)

optimizer = PoolMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=2)

model = TanimotoGPModel()
acq_func = EIAcquisition()
history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    observed_indices=init.observed_indices,
)

bo_loop = BOLoop(
    history=history,
    model=model,
    acq_func=acq_func,
    oracle=oracle,
    acqf_optimizer=optimizer,
    candidates=X,
)
bo_loop.run(n_iters=4)
```

### Generative BO

```python
# Coming soon
```


## Documentation

### Oracles

An oracle wraps the objective function and handles evaluation. Three main types are supported:

- `AnalyticOracle`: for continuous domains; takes a callable `f` and `bounds`
- `LookupOracle`: for fixed candidate pools; evaluates by indexing into a pre-featurized list of candidates
- `MolecularOracle`: for general molecular optimization; takes SMILES strings as input - subclass and implement `_evaluate(smiles) -> (N, 1) tensor`

```python
from molbo.oracle import AnalyticOracle
from molbo.oracle.lookup import LookupOracle

oracle = AnalyticOracle(f=f, dim=1, bounds=bounds)
oracle = LookupOracle(X_data=X, y_data=y)
```


### Featurization

BoTorch modules assume tensors as input, but molbo models are representation-agnostic by default. The choice of representation is up to the user and can be handled using a `Featurizer` object. Two featurizers are provided as conveniences:

```python
from molbo.model.featurizer import IdentityFeaturizer, MorganFingerprintFeaturizer

# IdentityFeaturizer: pass-through (default for all models)
IdentityFeaturizer()

# MorganFingerprintFeaturizer: maps SMILES strings to Morgan fingerprints
featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
X = featurizer(smiles)  # List[str] -> (N, 2048) tensor
```

In the **fixed-pool setting**, featurization is assumed upfront before constructing the oracle and candidate tensor. This avoids repeatedly featurizing the entire candidate pool every time the acquisition function is optimized. The model then operates on the pre-featurized inputs directly via `IdentityFeaturizer` (the default).

In the **generative setting**, one can use a featurizer to allow for a general representation:

```python
model = TanimotoGPModel(featurizer=MorganFingerprintFeaturizer())
```


### Surrogate models

Two existing models are implemented in molbo:

- `GPModel`: standard GP with RBF kernel; for continuous domains
- `TanimotoGPModel`: GP with Tanimoto kernel; for molecular fingerprints

Both default to `IdentityFeaturizer` and inputs are passed to the kernel unchanged.

```python
from molbo.model.gp import GPModel, TanimotoGPModel
from molbo.model.featurizer import MorganFingerprintFeaturizer

model = GPModel()
model = TanimotoGPModel()

# Generative setting: featurize SMILES automatically
model = TanimotoGPModel(featurizer=MorganFingerprintFeaturizer())
```

Additional models can be defined by subclassing the `SurrogateModel` ABC. Current models wrap a `SingleTaskGP` to interface with BoTorch acquisition functions (see [BoTorch documentation](https://botorch.org/docs/models#implementing-custom-models) for implementing custom models).


### Acquisition functions

Standard acquisition functions are available in `molbo.acquisition.base`. Some examples:

```python
from molbo.acquisition.base import LogEIAcquisition, UCBAcquisition, TSAcquisition

acq = LogEIAcquisition()
acq = UCBAcquisition(beta=1.0)
acq = TSAcquisition()
```

Batch acquisition functions (required for $q > 1$) are in `molbo.acquisition.monte_carlo`:

```python
from molbo.acquisition.monte_carlo import qLogEIAcquisition, qUCBAcquisition

acq = qLogEIAcquisition()
acq = qUCBAcquisition(beta=1.0)
```

Note: it is recommended to use `LogEIAcquisition` and `qLogEIAcquisition` ([ref](https://botorch.readthedocs.io/en/stable/acquisition.html#botorch.acquisition.analytic.ExpectedImprovement)).


### Acquisition optimizers

The acquisition optimizer determines how candidates are selected given the acquisition function.

**Continuous:**
```python
from molbo.acqf_optimizer.continuous import ContinuousMaximizer, ContinuousSampler

optimizer = ContinuousMaximizer(q=1)
optimizer = ContinuousMaximizer(q=4)  # batch; requires MC acquisition function
```

**Fixed-pool:**
```python
from molbo.acqf_optimizer.pool import PoolMaximizer, PoolSampler

optimizer = PoolMaximizer(q=1)
optimizer = PoolMaximizer(q=4)  # batch; requires MC acquisition function
optimizer = PoolSampler(q=1)    # samples proportional to acquisition values
```

**Generative:** subclass `AcqfOptimizer` and implement `_optimize` and `sample_init`.


### History

`History` tracks the state of a BO run - observations, metrics, logging, and checkpointing:

```python
from molbo.bo import History

history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    metrics=[my_metric],           # optional: List[Callable[[History], dict]]
    logger=WandBLogger(...),        # optional: logs metrics each iteration
    checkpoint_path="run.pt",      # optional: saves to disk every checkpoint_freq iters
    checkpoint_freq=10,
)
```

Custom metrics are callables of the form `f(history: History) -> dict`:

```python
def my_metric(history: History) -> dict:
    return {"metric_name": value}
```

To resume from a checkpoint:

```python
history = History.load("run.pt", metrics=[my_metric], logger=WandBLogger(...))
history = history or History(X_init=init.train_X, y_init=init.train_y, ...)
```

## Devices

The device is determined by the oracle:

```python
oracle = AnalyticOracle(...).to(device)
```

Training data inherits the oracle's device via `sample_init`, and the model follows from the training data.

## Data types

molbo does not set a global default dtype. However, GP fitting relies on `torch.float64` for numerical stability. Thus, all molbo components are expected to be set to `torch.float64`.
