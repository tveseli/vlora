#!/usr/bin/env python3
"""Real-world vLoRA benchmark with Lots-of-LoRAs adapters (Mistral-7B).

Downloads 8 diverse LoRA adapters from HuggingFace, runs the full 3-step
algorithm, and reports reconstruction error, variance explained, and
memory savings.

Usage:
    pip install vlora[hub]
    python examples/real_adapters.py
"""

from __future__ import annotations

import time

import torch

from vlora import SharedSubspace, load_adapter_from_hub
from vlora.ops import explained_variance_ratio

# ── Adapter selection ──────────────────────────────────────────────────
# 8 Lots-of-LoRAs adapters (Mistral-7B-Instruct-v0.2, rank 16, 4-bit)
# Chosen for task diversity: classification, QA, generation, summarization
ADAPTERS = [
    ("task581",  "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task581"),
    ("task909",  "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task909"),
    ("task132",  "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task132"),
    ("task1344", "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1344"),
    ("task1577", "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1577"),
    ("task1236", "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1236"),
    ("task785",  "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task785"),
    ("task172",  "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task172"),
]


def download_adapters():
    """Download all adapters from HuggingFace Hub."""
    print("=" * 60)
    print("Step 0: Downloading adapters from HuggingFace")
    print("=" * 60)
    adapters = []
    for name, repo_id in ADAPTERS:
        t0 = time.time()
        adapter = load_adapter_from_hub(repo_id)
        dt = time.time() - t0
        print(f"  {name}: {len(adapter.layer_names)} layers, rank {adapter.rank} ({dt:.1f}s)")
        adapters.append((name, adapter))
    print()
    return adapters


def relative_error(original, reconstructed, layer_names):
    """Compute relative L2 reconstruction error across all layers."""
    total_err_sq = 0.0
    total_norm_sq = 0.0
    for layer in layer_names:
        for side in ("lora_a", "lora_b"):
            orig = getattr(original, side)[layer].float()
            recon = getattr(reconstructed, side)[layer].float()
            total_err_sq += (orig - recon).norm().item() ** 2
            total_norm_sq += orig.norm().item() ** 2
    return (total_err_sq / max(total_norm_sq, 1e-12)) ** 0.5


def run_benchmark():
    """Run the full vLoRA benchmark."""
    named_adapters = download_adapters()
    names = [n for n, _ in named_adapters]
    adapters = [a for _, a in named_adapters]

    # Hold out the last adapter for projection test
    train_adapters = adapters[:-1]
    train_names = names[:-1]
    holdout_adapter = adapters[-1]
    holdout_name = names[-1]

    # ── Step 1: Build shared subspace (sweep k) ──────────────────────
    print("=" * 60)
    print("Step 1: Building shared subspace — variance explained vs. k")
    print("=" * 60)

    # Build with max components to get singular values
    full_sub = SharedSubspace.from_adapters(
        train_adapters,
        task_ids=train_names,
        num_components=len(train_adapters),  # max possible = N adapters
    )

    # Report variance explained per k
    print(f"\n  {'k':>3}  {'Var. Explained (A)':>18}  {'Var. Explained (B)':>18}")
    print(f"  {'─' * 3}  {'─' * 18}  {'─' * 18}")

    # Use the first layer as representative
    first_layer = full_sub.layer_names[0]
    sv_a = full_sub.singular_values_a[first_layer]
    sv_b = full_sub.singular_values_b[first_layer]
    cum_var_a = explained_variance_ratio(sv_a)
    cum_var_b = explained_variance_ratio(sv_b)

    for k in range(1, len(train_adapters) + 1):
        va = cum_var_a[k - 1].item() if k <= len(cum_var_a) else 1.0
        vb = cum_var_b[k - 1].item() if k <= len(cum_var_b) else 1.0
        print(f"  {k:>3}  {va:>18.4f}  {vb:>18.4f}")
    print()

    # ── Reconstruction error at different k values ───────────────────
    print("=" * 60)
    print("Reconstruction error vs. number of components (k)")
    print("=" * 60)
    print(f"\n  {'k':>3}  {'Mean Rel. Error':>15}  {'Max Rel. Error':>15}")
    print(f"  {'─' * 3}  {'─' * 15}  {'─' * 15}")

    for k in range(1, len(train_adapters) + 1):
        sub_k = SharedSubspace.from_adapters(
            train_adapters, task_ids=train_names, num_components=k
        )
        errors = []
        for tid, orig in zip(train_names, train_adapters):
            recon = sub_k.reconstruct(tid)
            errors.append(relative_error(orig, recon, sub_k.layer_names))
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        print(f"  {k:>3}  {mean_err:>15.6f}  {max_err:>15.6f}")
    print()

    # ── Step 2: Project held-out adapter ─────────────────────────────
    print("=" * 60)
    print(f"Step 2: Projecting held-out adapter ({holdout_name})")
    print("=" * 60)

    # Use k = len(train_adapters) for final subspace
    k_final = len(train_adapters)
    subspace = SharedSubspace.from_adapters(
        train_adapters, task_ids=train_names, num_components=k_final
    )

    proj = subspace.project(holdout_adapter, task_id=holdout_name)
    subspace.add_task(proj)
    recon_holdout = subspace.reconstruct(holdout_name)
    holdout_err = relative_error(holdout_adapter, recon_holdout, subspace.layer_names)
    print(f"\n  Held-out reconstruction error: {holdout_err:.6f}")
    print()

    # ── Step 3: Absorb held-out adapter ──────────────────────────────
    print("=" * 60)
    print(f"Step 3: Absorbing held-out adapter ({holdout_name})")
    print("=" * 60)

    subspace.absorb(holdout_adapter, holdout_name)
    recon_after = subspace.reconstruct(holdout_name)
    absorb_err = relative_error(holdout_adapter, recon_after, subspace.layer_names)
    print(f"\n  Post-absorb reconstruction error: {absorb_err:.6f}")
    print(f"  Total tasks in subspace: {len(subspace.tasks)}")
    print()

    # ── Memory comparison ────────────────────────────────────────────
    print("=" * 60)
    print("Memory comparison")
    print("=" * 60)

    n_adapters = len(ADAPTERS)
    n_layers = len(subspace.layer_names)
    rank = subspace.rank
    k = subspace.num_components

    # Per-adapter: all A and B matrices for all layers
    total_params_per_adapter = 0
    for layer in subspace.layer_names:
        a_shape = adapters[0].lora_a[layer].shape
        b_shape = adapters[0].lora_b[layer].shape
        total_params_per_adapter += a_shape.numel() + b_shape.numel()

    # Shared subspace: components + means (fixed cost) + loadings per task
    shared_params = 0
    loadings_per_task = 0
    for layer in subspace.layer_names:
        comp_a = subspace.components_a[layer]
        comp_b = subspace.components_b[layer]
        mean_a = subspace.means_a[layer]
        mean_b = subspace.means_b[layer]
        shared_params += comp_a.numel() + comp_b.numel() + mean_a.numel() + mean_b.numel()
        loadings_per_task += k * 2  # k loadings for A + k loadings for B

    bytes_per_param = 4  # float32

    print(f"\n  Adapters:           {n_adapters}")
    print(f"  Layers per adapter: {n_layers}")
    print(f"  LoRA rank:          {rank}")
    print(f"  Subspace components (k): {k}")
    print(f"  Params per adapter: {total_params_per_adapter:,}")
    print(f"  Shared basis params: {shared_params:,}")
    print(f"  Loadings per task:  {loadings_per_task:,}")

    # Show scaling: how compression improves with more adapters
    print(f"\n  {'N adapters':>12}  {'Full (MB)':>10}  {'vLoRA (MB)':>10}  {'Ratio':>8}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
    for n in [8, 50, 100, 500, 1000]:
        full = n * total_params_per_adapter
        vlora = shared_params + n * loadings_per_task
        full_mb = full * bytes_per_param / (1024 * 1024)
        vlora_mb = vlora * bytes_per_param / (1024 * 1024)
        ratio = full / vlora
        print(f"  {n:>12}  {full_mb:>10.1f}  {vlora_mb:>10.1f}  {ratio:>7.1f}×")
    print()

    # ── Save/load roundtrip ──────────────────────────────────────────
    print("=" * 60)
    print("Save/load roundtrip verification")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "subspace"
        subspace.save(save_path)

        loaded = SharedSubspace.load(save_path)

        # Verify reconstruction matches
        recon_before = subspace.reconstruct(train_names[0])
        recon_after_load = loaded.reconstruct(train_names[0])

        roundtrip_err = relative_error(recon_before, recon_after_load, subspace.layer_names)
        print(f"\n  Save/load roundtrip error: {roundtrip_err:.2e}")
        assert roundtrip_err < 1e-5, f"Roundtrip error too large: {roundtrip_err}"
        print("  ✓ Roundtrip verified (error < 1e-5)")

    print()
    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
