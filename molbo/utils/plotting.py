import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython.display import display
from ipywidgets import Button, HBox, IntSlider, VBox, interactive

from molbo.bo import BOLoop, BOMetrics

sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_curves(results_list: list[BOMetrics]):
    """
    TODO: Plot metric curves
    """
    pass


def plot_1d_iteration(
    model, X_grid, train_X, train_y, oracle, acq_func=None, new_X=None, new_y=None, axes=None
):
    """
    Plot a single BO iteration in 1D.

    Args:
        model: Fitted surrogate model
        X_grid: Grid points for plotting (N, 1)
        train_X: Observed X data up to this iteration (n, 1)
        train_y: Observed y data up to this iteration (n, 1)
        new_X: Next point(s) to query (B, 1) or None if final iteration
        new_y: Observed y at new_X (B, 1) or None
        oracle: Oracle object
        acq_func: Acquisition function (already updated with model)
    """
    mean, std = model(X_grid)

    # Subplots for model, acquisition function
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(6, 4))
        show_legend = True
    else:
        show_legend = False

    with torch.no_grad():
        ax = axes[0]

        # True function (no noise)
        ax.plot(X_grid, oracle._evaluate(X_grid), c="k", ls="--", label="Objective function")
        ax.plot(X_grid, mean, c="blue", label="Posterior mean")
        ax.fill_between(
            X_grid.squeeze(),
            mean.squeeze() - std.squeeze(),
            mean.squeeze() + std.squeeze(),
            color="blue",
            alpha=0.2,
            label="Posterior std",
        )
        # Plot (potentially noisy) observations
        ax.scatter(train_X, train_y, color="k", s=20, zorder=2, label="Observed data")
        if new_X is not None:
            ax.scatter(new_X, new_y, color="red", s=20, zorder=2, label="Pending observation")

        ax = axes[1]

        ax.plot(
            X_grid,
            acq_func(X_grid.reshape(-1, 1, 1)),
            color="limegreen",
            label="Acquisition function",
        )
        if new_X is not None:
            ax.scatter(
                new_X,
                acq_func(new_X.unsqueeze(-1)),
                marker="x",
                color="red",
                zorder=2,
            )

        if show_legend:
            fig.legend(bbox_to_anchor=(1.25, 0.7))
            plt.show()


def plot_1d_interactive(bo_loop: BOLoop):
    """Interactive slider to step through BO iterations in 1D."""

    history = bo_loop.history
    X_init = history["X_init"]
    y_init = history["y_init"]
    X_observed = history["X_observed"]
    y_observed = history["y_observed"]

    n_iters = history["iteration"][-1]

    # Create grid for plotting
    oracle = bo_loop.oracle
    X_grid = torch.linspace(
        oracle.bounds[0].item(),
        oracle.bounds[1].item(),
        int((oracle.bounds[1] - oracle.bounds[0]) * 20),
        dtype=torch.float64,
    ).unsqueeze(-1)

    # Pre-compute plot dependencies for smooth transitions
    models_list = []
    acq_funcs_list = []
    train_X_list = []
    train_y_list = []
    new_X_list = []
    new_y_list = []

    for i in range(n_iters + 1):
        # Slice data up to this iteration
        if i == 0:
            train_X = X_init
            train_y = y_init
            new_X = X_observed[0:1] if len(X_observed) > 0 else None
            new_y = y_observed[0:1] if len(y_observed) > 0 else None
        else:
            train_X = torch.cat([X_init, X_observed[:i]])
            train_y = torch.cat([y_init, y_observed[:i]])
            new_X = X_observed[i : i + 1] if i < n_iters else None
            new_y = y_observed[i : i + 1] if i < n_iters else None

        # Fit model
        import copy

        model = copy.deepcopy(bo_loop.model)
        model.initialize(train_X, train_y)
        model.fit()
        models_list.append(model)

        # Get acquisition values
        acq_func_class = type(bo_loop.acq_func)
        acq_func = acq_func_class()
        acq_func.update(model)
        acq_funcs_list.append(acq_func)

        train_X_list.append(train_X)
        train_y_list.append(train_y)
        new_X_list.append(new_X)
        new_y_list.append(new_y)

    def step(iteration):
        model = models_list[iteration]
        acq_func = acq_funcs_list[iteration]
        train_X = train_X_list[iteration]
        train_y = train_y_list[iteration]
        new_X = new_X_list[iteration]
        new_y = new_y_list[iteration]

        # Plot
        plot_1d_iteration(model, X_grid, train_X, train_y, oracle, acq_func, new_X, new_y)

    # Create interactive widget
    slider = IntSlider(min=0, max=n_iters, step=1, value=0, description="Iteration:")
    w = interactive(step, iteration=slider)

    # Create buttons
    prev_button = Button(description="◀")
    next_button = Button(description="▶")

    def on_prev(b):
        if slider.value > 0:
            slider.value -= 1

    def on_next(b):
        if slider.value < n_iters:
            slider.value += 1

    prev_button.on_click(on_prev)
    next_button.on_click(on_next)

    # Display
    controls = HBox([prev_button, slider, next_button])
    display(VBox([controls, w.children[-1]]))


def plot_discrete_iteration(
    model, X_data, y_data, train_X, train_y, new_X, new_y, acq_func, axes=None
):

    sorted_indices = y_data.squeeze().argsort()
    X_sorted = X_data[sorted_indices]
    y_sorted = y_data.squeeze()[sorted_indices]
    xs = torch.arange(len(y_sorted))

    matches = (X_sorted.unsqueeze(1) == train_X.unsqueeze(0)).all(
        dim=-1
    )  # (N, n); find indices of X where a training data point matches
    observed_mask = matches.any(dim=-1)  # (N,) which values have a "True" element
    observed_xs = xs[observed_mask]
    observed_ys = train_y.squeeze()[matches[observed_mask].int().argmax(dim=-1)]

    if new_X is not None:
        new_match = (X_sorted == new_X).all(dim=-1)
        new_xs = xs[new_match]
        new_ys = new_y.squeeze()

    with torch.no_grad():
        mean, std = model(X_sorted)
        mean = mean.squeeze()
        std = std.squeeze()
        acq_vals = acq_func(X_sorted.unsqueeze(1)).squeeze()

    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(6, 4))
        show_legend = True
    else:
        show_legend = False

    ax = axes[0]
    ax.plot(xs, y_sorted, c="k", ls="--", label="Objective")
    ax.errorbar(xs, mean, yerr=std, fmt="none", color="blue", alpha=0.3, label="Posterior std")
    ax.scatter(observed_xs, observed_ys, color="k", s=20, zorder=3, label="Observed")
    if new_X is not None:
        ax.scatter(new_xs, new_ys, color="red", s=20, zorder=4, label="Pending observation")

    ax = axes[1]
    ax.bar(xs, acq_vals.detach(), color="limegreen", label="Acquisition")
    if new_X is not None:
        ax.bar(new_xs, acq_vals[new_match].detach(), color="red")

    if show_legend:
        fig.legend(bbox_to_anchor=(1.25, 0.65))
        plt.show()
