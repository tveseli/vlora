"""SharedSubspace — core state container and 3-step algorithm.

Step 1: from_adapters  — build shared basis via SVD
Step 2: project        — project new adapter onto basis
Step 3: absorb         — incorporate new adapter, recompute basis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from vlora.io import LoRAWeights, stack_lora_weights
from vlora.ops import (
    compute_svd,
    explained_variance_ratio,
    gram_schmidt,
    project_onto_subspace,
    reconstruct_from_subspace,
    select_num_components,
)


@dataclass
class TaskProjection:
    """A single task's representation in the shared subspace."""

    task_id: str
    loadings_a: dict[str, Tensor]  # layer_name -> (k,)
    loadings_b: dict[str, Tensor]  # layer_name -> (k,)


class SharedSubspace:
    """Shared low-rank subspace for LoRA adapters.

    Maintains per-layer orthonormal basis vectors (components) and
    per-task coefficient vectors (loadings).
    """

    def __init__(
        self,
        layer_names: list[str],
        components_a: dict[str, Tensor],
        components_b: dict[str, Tensor],
        singular_values_a: dict[str, Tensor],
        singular_values_b: dict[str, Tensor],
        means_a: dict[str, Tensor],
        means_b: dict[str, Tensor],
        tasks: dict[str, TaskProjection],
        rank: int,
        num_components: int,
    ):
        self.layer_names = layer_names
        self.components_a = components_a
        self.components_b = components_b
        self.singular_values_a = singular_values_a
        self.singular_values_b = singular_values_b
        self.means_a = means_a
        self.means_b = means_b
        self.tasks = tasks
        self.rank = rank
        self.num_components = num_components

    @classmethod
    def from_adapters(
        cls,
        adapters: list[LoRAWeights],
        task_ids: list[str] | None = None,
        variance_threshold: float = 0.6,
        num_components: int | None = None,
    ) -> SharedSubspace:
        """Step 1: Build shared subspace from existing adapters.

        Stacks each adapter's flattened weights, runs SVD per layer,
        and projects all adapters onto the resulting basis.

        Args:
            adapters: List of LoRA adapters to initialize from.
            task_ids: Names for each adapter. Defaults to "task_0", "task_1", etc.
            variance_threshold: Minimum cumulative variance to explain (used if
                num_components is None).
            num_components: Explicit number of basis vectors per layer.
                Overrides variance_threshold if set.
        """
        if not adapters:
            raise ValueError("Need at least one adapter")

        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(adapters))]
        if len(task_ids) != len(adapters):
            raise ValueError("task_ids length must match adapters length")

        layer_names = adapters[0].layer_names
        rank = adapters[0].rank

        # Stack weights into data matrices
        stacked_a = stack_lora_weights(adapters, side="A")
        stacked_b = stack_lora_weights(adapters, side="B")

        components_a: dict[str, Tensor] = {}
        components_b: dict[str, Tensor] = {}
        sv_a: dict[str, Tensor] = {}
        sv_b: dict[str, Tensor] = {}
        means_a: dict[str, Tensor] = {}
        means_b: dict[str, Tensor] = {}

        resolved_k: int | None = None

        for layer in layer_names:
            for side, stacked, comp_dict, sv_dict, mean_dict in [
                ("A", stacked_a, components_a, sv_a, means_a),
                ("B", stacked_b, components_b, sv_b, means_b),
            ]:
                data = stacked[layer]
                comps, svals, mean = compute_svd(data, num_components=None, center=True)

                if num_components is not None:
                    k = min(num_components, len(svals))
                else:
                    k = select_num_components(svals, variance_threshold)

                if resolved_k is None:
                    resolved_k = k
                # Use consistent k across layers for simplicity
                k = resolved_k

                comp_dict[layer] = comps[:k]
                sv_dict[layer] = svals[:k]
                mean_dict[layer] = mean

        resolved_k = resolved_k or 1

        # Project all input adapters onto the basis
        tasks: dict[str, TaskProjection] = {}
        for i, (adapter, tid) in enumerate(zip(adapters, task_ids)):
            loadings_a: dict[str, Tensor] = {}
            loadings_b: dict[str, Tensor] = {}
            for layer in layer_names:
                wa = adapter.lora_a[layer].flatten() - means_a[layer]
                wb = adapter.lora_b[layer].flatten() - means_b[layer]
                loadings_a[layer] = project_onto_subspace(wa, components_a[layer])
                loadings_b[layer] = project_onto_subspace(wb, components_b[layer])
            tasks[tid] = TaskProjection(
                task_id=tid, loadings_a=loadings_a, loadings_b=loadings_b
            )

        return cls(
            layer_names=layer_names,
            components_a=components_a,
            components_b=components_b,
            singular_values_a=sv_a,
            singular_values_b=sv_b,
            means_a=means_a,
            means_b=means_b,
            tasks=tasks,
            rank=rank,
            num_components=resolved_k,
        )

    def project(self, adapter: LoRAWeights, task_id: str) -> TaskProjection:
        """Step 2a: Project a new adapter onto the existing basis."""
        loadings_a: dict[str, Tensor] = {}
        loadings_b: dict[str, Tensor] = {}

        for layer in self.layer_names:
            wa = adapter.lora_a[layer].flatten() - self.means_a[layer]
            wb = adapter.lora_b[layer].flatten() - self.means_b[layer]
            loadings_a[layer] = project_onto_subspace(wa, self.components_a[layer])
            loadings_b[layer] = project_onto_subspace(wb, self.components_b[layer])

        return TaskProjection(
            task_id=task_id, loadings_a=loadings_a, loadings_b=loadings_b
        )

    def add_task(self, projection: TaskProjection) -> None:
        """Register a projected task in the subspace."""
        self.tasks[projection.task_id] = projection

    def reconstruct(self, task_id: str) -> LoRAWeights:
        """Reconstruct full LoRA weights for a task from its loadings."""
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")

        proj = self.tasks[task_id]
        lora_a: dict[str, Tensor] = {}
        lora_b: dict[str, Tensor] = {}

        for layer in self.layer_names:
            flat_a = reconstruct_from_subspace(
                self.components_a[layer], proj.loadings_a[layer]
            ) + self.means_a[layer]
            flat_b = reconstruct_from_subspace(
                self.components_b[layer], proj.loadings_b[layer]
            ) + self.means_b[layer]

            # Recover original matrix shapes from the adapter's rank
            # A: (rank, in_features), B: (out_features, rank)
            ref_a_shape = (self.rank, flat_a.numel() // self.rank)
            ref_b_shape = (flat_b.numel() // self.rank, self.rank)
            lora_a[layer] = flat_a.reshape(ref_a_shape)
            lora_b[layer] = flat_b.reshape(ref_b_shape)

        return LoRAWeights(
            layer_names=self.layer_names,
            lora_a=lora_a,
            lora_b=lora_b,
            rank=self.rank,
        )

    def absorb(self, new_adapter: LoRAWeights, new_task_id: str) -> None:
        """Step 3: Absorb a new adapter, recomputing the shared basis.

        Reconstructs all existing tasks, adds the new adapter, then
        reruns SVD to produce an updated basis.
        """
        # Reconstruct all existing tasks as full adapters
        all_adapters = []
        all_ids = []
        for tid, _ in self.tasks.items():
            all_adapters.append(self.reconstruct(tid))
            all_ids.append(tid)

        all_adapters.append(new_adapter)
        all_ids.append(new_task_id)

        # Rebuild subspace from scratch
        new_sub = SharedSubspace.from_adapters(
            all_adapters,
            task_ids=all_ids,
            num_components=self.num_components,
        )

        # Update self in-place
        self.layer_names = new_sub.layer_names
        self.components_a = new_sub.components_a
        self.components_b = new_sub.components_b
        self.singular_values_a = new_sub.singular_values_a
        self.singular_values_b = new_sub.singular_values_b
        self.means_a = new_sub.means_a
        self.means_b = new_sub.means_b
        self.tasks = new_sub.tasks
        self.num_components = new_sub.num_components

    def get_trainable_params(
        self, task_id: str, num_expand: int = 0
    ) -> dict[str, Tensor]:
        """Get trainable loading parameters for a task.

        Useful for integrating with a training loop: freeze the components,
        train only the loadings.

        Args:
            task_id: Task whose loadings to return.
            num_expand: Number of extra orthogonal directions to add via
                Gram-Schmidt (gives the optimizer room to escape the subspace).

        Returns:
            Dict of parameter name -> tensor (with requires_grad=True).
        """
        if num_expand > 0:
            for layer in self.layer_names:
                random_a = torch.randn(num_expand, self.components_a[layer].shape[1])
                random_b = torch.randn(num_expand, self.components_b[layer].shape[1])
                self.components_a[layer] = gram_schmidt(self.components_a[layer], random_a)
                self.components_b[layer] = gram_schmidt(self.components_b[layer], random_b)

            # Re-project the task onto the expanded basis
            proj = self.tasks.get(task_id)
            if proj is not None:
                for layer in self.layer_names:
                    old_k_a = proj.loadings_a[layer].shape[0]
                    new_k_a = self.components_a[layer].shape[0]
                    if new_k_a > old_k_a:
                        proj.loadings_a[layer] = torch.cat([
                            proj.loadings_a[layer],
                            torch.zeros(new_k_a - old_k_a),
                        ])
                    old_k_b = proj.loadings_b[layer].shape[0]
                    new_k_b = self.components_b[layer].shape[0]
                    if new_k_b > old_k_b:
                        proj.loadings_b[layer] = torch.cat([
                            proj.loadings_b[layer],
                            torch.zeros(new_k_b - old_k_b),
                        ])

        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")

        params = {}
        proj = self.tasks[task_id]
        for layer in self.layer_names:
            la = proj.loadings_a[layer].clone().detach().requires_grad_(True)
            lb = proj.loadings_b[layer].clone().detach().requires_grad_(True)
            params[f"{layer}.loadings_a"] = la
            params[f"{layer}.loadings_b"] = lb

        return params

    def save(self, path: str | Path) -> None:
        """Serialize the subspace to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save components and means (contiguous() needed for safetensors)
        tensors = {}
        for layer in self.layer_names:
            tensors[f"{layer}.components_a"] = self.components_a[layer].contiguous()
            tensors[f"{layer}.components_b"] = self.components_b[layer].contiguous()
            tensors[f"{layer}.sv_a"] = self.singular_values_a[layer].contiguous()
            tensors[f"{layer}.sv_b"] = self.singular_values_b[layer].contiguous()
            tensors[f"{layer}.mean_a"] = self.means_a[layer].contiguous()
            tensors[f"{layer}.mean_b"] = self.means_b[layer].contiguous()

        save_file(tensors, str(path / "subspace.safetensors"))

        # Save per-task loadings
        for tid, proj in self.tasks.items():
            task_tensors = {}
            for layer in self.layer_names:
                task_tensors[f"{layer}.loadings_a"] = proj.loadings_a[layer].contiguous()
                task_tensors[f"{layer}.loadings_b"] = proj.loadings_b[layer].contiguous()
            save_file(task_tensors, str(path / f"task_{tid}.safetensors"))

        # Save metadata
        import json
        meta = {
            "layer_names": self.layer_names,
            "task_ids": list(self.tasks.keys()),
            "rank": self.rank,
            "num_components": self.num_components,
        }
        with open(path / "subspace_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SharedSubspace:
        """Deserialize a subspace from disk."""
        import json

        path = Path(path)

        with open(path / "subspace_meta.json") as f:
            meta = json.load(f)

        layer_names = meta["layer_names"]
        task_ids = meta["task_ids"]
        rank = meta["rank"]
        num_components = meta["num_components"]

        tensors = load_file(str(path / "subspace.safetensors"))
        components_a = {l: tensors[f"{l}.components_a"] for l in layer_names}
        components_b = {l: tensors[f"{l}.components_b"] for l in layer_names}
        sv_a = {l: tensors[f"{l}.sv_a"] for l in layer_names}
        sv_b = {l: tensors[f"{l}.sv_b"] for l in layer_names}
        means_a = {l: tensors[f"{l}.mean_a"] for l in layer_names}
        means_b = {l: tensors[f"{l}.mean_b"] for l in layer_names}

        tasks = {}
        for tid in task_ids:
            task_tensors = load_file(str(path / f"task_{tid}.safetensors"))
            loadings_a = {l: task_tensors[f"{l}.loadings_a"] for l in layer_names}
            loadings_b = {l: task_tensors[f"{l}.loadings_b"] for l in layer_names}
            tasks[tid] = TaskProjection(
                task_id=tid, loadings_a=loadings_a, loadings_b=loadings_b
            )

        return cls(
            layer_names=layer_names,
            components_a=components_a,
            components_b=components_b,
            singular_values_a=sv_a,
            singular_values_b=sv_b,
            means_a=means_a,
            means_b=means_b,
            tasks=tasks,
            rank=rank,
            num_components=num_components,
        )
