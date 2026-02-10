"""VLoRAModel — inference wrapper that applies reconstructed LoRA deltas."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from vlora.subspace import SharedSubspace


class VLoRAModel(nn.Module):
    """Wraps a base model with a shared subspace for multi-task LoRA inference.

    Reconstructs task-specific LoRA deltas on demand and applies them to
    the base model's linear layers during forward pass.

    Usage:
        subspace = SharedSubspace.load("shared_subspace/")
        base_model = AutoModelForCausalLM.from_pretrained("model-name")
        model = VLoRAModel(base_model, subspace)

        model.set_task("task_0")
        output = model(input_ids)

        model.set_task("task_1")  # switches adapter, cached if same task
        output = model(input_ids)
    """

    def __init__(
        self,
        base_model: nn.Module,
        subspace: SharedSubspace,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.subspace = subspace
        self.scaling = scaling
        self._active_task: str | None = None
        self._cached_deltas: dict[str, Tensor] | None = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def set_task(self, task_id: str) -> None:
        """Set the active task adapter. Reconstructs and caches if changed."""
        if task_id == self._active_task:
            return

        if task_id not in self.subspace.tasks:
            available = ", ".join(sorted(self.subspace.tasks.keys()))
            raise KeyError(f"Unknown task '{task_id}'. Available: {available}")

        # Reconstruct and cache the LoRA deltas
        weights = self.subspace.reconstruct(task_id)
        self._cached_deltas = {}
        for layer_name in weights.layer_names:
            # delta_W = B @ A
            delta = weights.lora_b[layer_name] @ weights.lora_a[layer_name]
            self._cached_deltas[layer_name] = delta

        self._active_task = task_id
        self._apply_hooks()

    def clear_task(self) -> None:
        """Remove the active task adapter."""
        self._remove_hooks()
        self._active_task = None
        self._cached_deltas = None

    def _apply_hooks(self) -> None:
        """Register forward hooks on matching linear layers."""
        self._remove_hooks()

        if self._cached_deltas is None:
            return

        for name, module in self.base_model.named_modules():
            if name in self._cached_deltas and isinstance(module, nn.Linear):
                delta = self._cached_deltas[name]
                hook = module.register_forward_hook(
                    self._make_lora_hook(delta)
                )
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_lora_hook(self, delta: Tensor):
        """Create a forward hook that adds LoRA delta to the output."""
        scaling = self.scaling

        def hook(module: nn.Module, input: Any, output: Tensor) -> Tensor:
            # input[0] is the input tensor to the linear layer
            x = input[0] if isinstance(input, tuple) else input
            lora_out = x @ delta.T.to(x.device, x.dtype)
            return output + scaling * lora_out

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass through the base model with active LoRA adapter."""
        return self.base_model(*args, **kwargs)

    @property
    def active_task(self) -> str | None:
        """Currently active task ID, or None."""
        return self._active_task

    @property
    def available_tasks(self) -> list[str]:
        """List of available task IDs."""
        return sorted(self.subspace.tasks.keys())

    def reconstruct_state_dict(self, task_id: str) -> dict[str, Tensor]:
        """Get the LoRA delta weight dict for a task without applying hooks.

        Returns dict of {layer_name: delta_W} where delta_W = B @ A.
        Useful for manual integration with custom model architectures.
        """
        weights = self.subspace.reconstruct(task_id)
        deltas = {}
        for layer_name in weights.layer_names:
            deltas[layer_name] = weights.lora_b[layer_name] @ weights.lora_a[layer_name]
        return deltas
