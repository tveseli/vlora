"""Pure math operations for shared low-rank subspace computation.

All functions are stateless and operate on raw tensors — no adapter
or subspace concepts leak in here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_svd(
    data_matrix: Tensor,
    num_components: int | None = None,
    center: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """SVD on data matrix, returning top-k components.

    Args:
        data_matrix: (N, D) matrix where rows are observations.
        num_components: Number of singular vectors to keep. If None, keep all.
        center: Whether to mean-center rows before SVD.

    Returns:
        components: (k, D) right singular vectors (shared basis).
        singular_values: (k,) corresponding singular values.
        mean: (D,) row mean (zeros if center=False).
    """
    if center:
        mean = data_matrix.mean(dim=0)
        centered = data_matrix - mean
    else:
        mean = torch.zeros(data_matrix.shape[1], dtype=data_matrix.dtype, device=data_matrix.device)
        centered = data_matrix

    # full_matrices=False gives the economy SVD — U is (N, min(N,D))
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    k = num_components if num_components is not None else len(S)
    k = min(k, len(S))

    return Vh[:k], S[:k], mean


def project_onto_subspace(weights: Tensor, components: Tensor) -> Tensor:
    """Project weight vectors onto the subspace basis.

    Args:
        weights: (D,) or (N, D) weight vectors to project.
        components: (k, D) orthonormal basis vectors.

    Returns:
        loadings: (k,) or (N, k) projection coefficients.
    """
    # loadings = weights @ V^T  (since components = V^T, i.e. rows are basis vectors)
    return weights @ components.T


def reconstruct_from_subspace(components: Tensor, loadings: Tensor) -> Tensor:
    """Reconstruct weight vectors from subspace loadings.

    Args:
        components: (k, D) orthonormal basis vectors.
        loadings: (k,) or (N, k) projection coefficients.

    Returns:
        reconstructed: (D,) or (N, D) reconstructed weight vectors.
    """
    return loadings @ components


def gram_schmidt(basis: Tensor, new_vectors: Tensor) -> Tensor:
    """Orthogonalize new_vectors against an existing orthonormal basis.

    Appends only those new directions that have non-trivial norm after
    projection removal (threshold: 1e-6).

    Args:
        basis: (k, D) existing orthonormal basis.
        new_vectors: (m, D) candidate vectors.

    Returns:
        expanded_basis: (k + n, D) where n <= m new orthogonal directions
            were found.
    """
    vectors = list(basis)

    for v in new_vectors:
        v = v.clone()
        # Remove components along every existing basis vector
        for b in vectors:
            v = v - (v @ b) * b
        norm = v.norm()
        if norm > 1e-6:
            vectors.append(v / norm)

    return torch.stack(vectors)


def explained_variance_ratio(singular_values: Tensor) -> Tensor:
    """Compute cumulative explained variance ratio from singular values.

    Args:
        singular_values: (k,) singular values in descending order.

    Returns:
        cumulative_ratio: (k,) cumulative fraction of total variance explained.
    """
    variance = singular_values ** 2
    total = variance.sum()
    if total == 0:
        return torch.zeros_like(variance)
    return torch.cumsum(variance, dim=0) / total


def select_num_components(singular_values: Tensor, threshold: float = 0.6) -> int:
    """Select number of components to explain at least `threshold` variance.

    Args:
        singular_values: (k,) singular values in descending order.
        threshold: Minimum cumulative variance ratio (default 0.6 per paper).

    Returns:
        Number of components needed.
    """
    cumulative = explained_variance_ratio(singular_values)
    # Find first index where cumulative ratio >= threshold
    above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return len(singular_values)
    return above[0].item() + 1
