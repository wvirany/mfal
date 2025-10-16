import torch
from botorch.acquisition import LogExpectedImprovement


def select_next_molecule(model, train_y, embeddings, queried_indices):
    """
    Select next molecule using expected improvement.

    Args:
        model: Trained GP model
        train_y: (n, 1) tensor of observed scores
        embeddings: (N, d) tensor of all molecule embeddings
        queried_indices: (n,) tensor of indices of already queried points

    Returns:
        next_idx: Index of molecule with highest EI
    """

    # Get unqueried indices
    all_indices = set(range(len(embeddings)))
    unqueried_indices = list(all_indices - set(queried_indices))

    if len(unqueried_indices) == 0:
        raise ValueError("No unqueried points left")

    unqueried_embeddings = embeddings[unqueried_indices]  # (n_unqueried, d)

    # Get EI for unqueried embeddings
    best_f = train_y.max().item()
    EI = LogExpectedImprovement(model=model, best_f=best_f)

    # Evaluate EI for unqueried points
    unqueried_batch = unqueried_embeddings.unsqueeze(1)  # Add batch dimension: (n_unqueried, 1, d)
    with torch.no_grad():
        ei_values = EI(unqueried_batch)  # (n_unqueried,)

    # Select point with highest EI
    best_idx = torch.argmax(ei_values).item()
    next_idx = unqueried_indices[best_idx]

    return next_idx
