"""End-to-end example of the vLoRA 3-step pipeline.

This example uses synthetic adapters to demonstrate the workflow.
Replace the adapter creation with `vlora.load_adapter(path)` for real use.
"""

import torch

from vlora import LoRAWeights, SharedSubspace


def make_synthetic_adapter(
    layer_names: list[str],
    rank: int = 8,
    in_features: int = 512,
    out_features: int = 512,
    task_id: str = "synthetic",
) -> LoRAWeights:
    """Create a random LoRA adapter for demonstration."""
    lora_a = {name: torch.randn(rank, in_features) * 0.01 for name in layer_names}
    lora_b = {name: torch.randn(out_features, rank) * 0.01 for name in layer_names}
    return LoRAWeights(
        layer_names=layer_names,
        lora_a=lora_a,
        lora_b=lora_b,
        rank=rank,
    )


def main():
    layers = ["layers.0.q_proj", "layers.0.v_proj", "layers.1.q_proj", "layers.1.v_proj"]
    rank = 8

    # Create some synthetic adapters (in practice, load real PEFT adapters)
    print("Creating 5 synthetic adapters...")
    adapters = [make_synthetic_adapter(layers, rank=rank, task_id=f"task_{i}") for i in range(5)]
    task_ids = [f"task_{i}" for i in range(5)]

    # ── Step 1: Build shared subspace ──
    print("\n── Step 1: Initialize subspace ──")
    subspace = SharedSubspace.from_adapters(adapters, task_ids=task_ids, num_components=3)
    print(f"  Subspace has {subspace.num_components} components across {len(subspace.layer_names)} layers")
    print(f"  Tasks registered: {list(subspace.tasks.keys())}")

    # ── Step 2: Project a new adapter ──
    print("\n── Step 2: Project new adapter ──")
    new_adapter = make_synthetic_adapter(layers, rank=rank, task_id="new_task")
    projection = subspace.project(new_adapter, task_id="new_task")
    subspace.add_task(projection)
    print(f"  Projected 'new_task' — loadings shape: {projection.loadings_a[layers[0]].shape}")

    # Check reconstruction quality
    reconstructed = subspace.reconstruct("new_task")
    for layer in layers[:1]:
        orig = new_adapter.lora_a[layer].flatten()
        recon = reconstructed.lora_a[layer].flatten()
        error = (orig - recon).norm() / orig.norm()
        print(f"  Reconstruction error ({layer} A): {error:.4f}")

    # ── Step 3: Absorb new adapter ──
    print("\n── Step 3: Absorb new adapter ──")
    another_adapter = make_synthetic_adapter(layers, rank=rank, task_id="absorbed")
    subspace.absorb(another_adapter, new_task_id="absorbed")
    print(f"  Tasks after absorb: {list(subspace.tasks.keys())}")

    # Verify round-trip
    for tid in ["task_0", "absorbed"]:
        recon = subspace.reconstruct(tid)
        print(f"  Reconstructed '{tid}' — A shape: {recon.lora_a[layers[0]].shape}")

    # ── Serialization ──
    print("\n── Save & Load ──")
    subspace.save("/tmp/vlora_demo_subspace")
    loaded = SharedSubspace.load("/tmp/vlora_demo_subspace")
    print(f"  Loaded subspace with {len(loaded.tasks)} tasks")

    print("\nDone!")


if __name__ == "__main__":
    main()
