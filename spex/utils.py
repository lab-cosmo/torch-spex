import torch


def compute_distance(R):
    """Compute distance over last dimension.

    Args:
        R (Tensor): Input tensor of shape ``[*dims, 3]``.

    Returns:
        Tensor of distances of shape ``[*dims]``.

    """
    # todo: make safe
    return torch.linalg.vector_norm(R, dim=-1)
