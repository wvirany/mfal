import torch
from botorch.acquisition import LogExpectedImprovement


def select_next_molecule(model, train_y, embeddings, queried_indices, batch_size: int = 1000):
    """
    Select next molecule using expected improvement.

    Args:
        model: Trained GP model
        train_y: (n, 1) tensor of observed scores
        embeddings: (N, d) tensor of all molecule embeddings
        queried_indices: (n,) tensor of indices of already queried points
        batch_size: Number of candidates to evaluate at once

    Returns:
        next_idx: Index of molecule with highest EI.
    """
    device = next(model.parameters()).device  # Get model device

    # Get unqueried indices
    all_indices = set(range(len(embeddings)))
    unqueried_indices = list(all_indices - set(queried_indices))

    if len(unqueried_indices) == 0:
        raise ValueError("No unqueried points left")

    # Get EI for unqueried embeddings
    best_f = train_y.max().item()
    EI = LogExpectedImprovement(model=model, best_f=best_f)

    # Evaluate EI for unqueried points in batches
    ei_values = []
    for i in range(0, len(unqueried_indices), batch_size):
        batch_indices = unqueried_indices[i : i + batch_size]
        batch_embeddings = embeddings[batch_indices].to(
            device
        )  # Move batch to GPU, shape: (batch_size, d)

        # Add batch dimension: (batch_size, 1, d)
        batch_embeddings = batch_embeddings.unsqueeze(1)

        with torch.no_grad():
            batch_ei = EI(batch_embeddings)  # (batch_size,)
            ei_values.append(batch_ei.cpu())  # Move EI values back to CPU

        # Clean up GPU memory
        del batch_embeddings

    # Concatenate all EI values
    ei_values = torch.cat(ei_values)  # (n_unqueried,)

    # Select point with highest EI
    best_idx = torch.argmax(ei_values).item()
    next_idx = unqueried_indices[best_idx]

    return next_idx
