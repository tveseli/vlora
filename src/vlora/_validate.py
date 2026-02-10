"""Input validation helpers for vlora public APIs."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from vlora.io import LoRAWeights
    from vlora.subspace import SharedSubspace

logger = logging.getLogger("vlora")


def check_adapters_compatible(adapters: list[LoRAWeights]) -> None:
    """Validate that adapters can be used together (same rank, overlapping layers)."""
    if not adapters:
        raise ValueError("Need at least one adapter.")

    ranks = {a.rank for a in adapters}
    if len(ranks) > 1:
        raise ValueError(
            f"Adapters have inconsistent ranks: {sorted(ranks)}. "
            "All adapters must have the same LoRA rank to share a subspace."
        )


def check_adapter_matches_subspace(
    adapter: LoRAWeights, subspace: SharedSubspace, operation: str = "operation"
) -> None:
    """Validate that an adapter is compatible with a subspace."""
    if adapter.rank != subspace.rank:
        raise ValueError(
            f"Cannot {operation}: adapter rank ({adapter.rank}) does not match "
            f"subspace rank ({subspace.rank})."
        )

    common = set(adapter.layer_names) & set(subspace.layer_names)
    if not common:
        raise ValueError(
            f"Cannot {operation}: adapter and subspace share no common layers. "
            f"Adapter layers: {adapter.layer_names[:3]}... "
            f"Subspace layers: {subspace.layer_names[:3]}..."
        )

    missing = set(subspace.layer_names) - set(adapter.layer_names)
    if missing:
        warnings.warn(
            f"Adapter is missing {len(missing)} subspace layers "
            f"(e.g. {sorted(missing)[:2]}). These will use mean values.",
            stacklevel=3,
        )


def check_task_exists(subspace: SharedSubspace, task_id: str) -> None:
    """Raise KeyError with helpful message if task not found."""
    if task_id not in subspace.tasks:
        available = ", ".join(sorted(subspace.tasks.keys()))
        raise KeyError(
            f"Unknown task '{task_id}'. "
            f"Available tasks: [{available}]. "
            f"Use subspace.tasks.keys() to list all tasks."
        )


def check_tensor_health(tensor: Tensor, name: str = "tensor") -> None:
    """Check for NaN/Inf in a tensor."""
    if torch.isnan(tensor).any():
        raise ValueError(
            f"NaN detected in {name}. This usually indicates numerical "
            "instability during SVD. Try using fewer components or "
            "checking your input adapters for degenerate weights."
        )
    if torch.isinf(tensor).any():
        raise ValueError(
            f"Inf detected in {name}. This usually indicates overflow "
            "during computation. Try using float32 precision."
        )
