# vLoRA

**Shared low-rank subspaces for efficient LoRA adapter management.**

Based on the [Share paper](https://arxiv.org/abs/2602.06043): LoRA adapters across tasks share a common low-rank subspace. Instead of storing *N* separate adapters, maintain **one shared basis** and **per-task coefficient vectors** — achieving up to 100× parameter reduction.

## Install

```bash
pip install vlora
```

Or from source:
```bash
git clone https://github.com/tveseli/vlora.git
cd vlora
pip install -e ".[dev]"
```

## Quickstart

```python
from vlora import SharedSubspace, load_adapter

# Step 1: Build shared subspace from existing adapters
adapters = [load_adapter(f"adapters/task_{i}") for i in range(5)]
subspace = SharedSubspace.from_adapters(adapters, num_components=16)

# Step 2: Project a new adapter (only stores small loadings vector)
new_adapter = load_adapter("adapters/new_task")
projection = subspace.project(new_adapter, task_id="new_task")
subspace.add_task(projection)

# Step 3: Absorb — recompute basis to include new adapter
subspace.absorb(load_adapter("adapters/another_task"), new_task_id="another")

# Reconstruct any task back to full LoRA weights
weights = subspace.reconstruct("new_task")

# Save / load
subspace.save("shared_subspace/")
subspace = SharedSubspace.load("shared_subspace/")
```

## The 3-Step Algorithm

| Step | Method | What happens |
|------|--------|-------------|
| **1. Initialize** | `SharedSubspace.from_adapters()` | SVD on stacked weight matrices → shared basis |
| **2. Project** | `subspace.project()` | New adapter → small loadings vector |
| **3. Absorb** | `subspace.absorb()` | Incorporate new adapter, recompute basis |

## API Reference

### Core

- **`SharedSubspace`** — Central state container. Holds per-layer basis and per-task loadings.
  - `.from_adapters(adapters, ...)` — Build from existing adapters
  - `.project(adapter, task_id)` → `TaskProjection`
  - `.add_task(projection)` — Register a projected task
  - `.reconstruct(task_id)` → `LoRAWeights`
  - `.absorb(adapter, task_id)` — Incorporate + recompute
  - `.get_trainable_params(task_id)` — For training integration
  - `.save(path)` / `.load(path)` — Serialization

### I/O

- **`load_adapter(path)`** — Load PEFT adapter from disk (safetensors)
- **`load_adapter_from_hub(repo_id)`** — Load from HuggingFace Hub
- **`save_adapter(weights, path)`** — Save back to PEFT format

### Pipeline (convenience)

- **`init_subspace(paths, ...)`** — Load + build in one call
- **`absorb_task(subspace, path, task_id)`** — Load + absorb
- **`extract_adapter(subspace, task_id, path)`** — Reconstruct + save

### Math ops

- `compute_svd`, `project_onto_subspace`, `reconstruct_from_subspace`
- `gram_schmidt`, `explained_variance_ratio`, `select_num_components`

## Dependencies

- `torch >= 2.0`
- `safetensors >= 0.4`
- `huggingface-hub >= 0.20` *(optional, for Hub loading)*

## Citation

```bibtex
@article{share2025,
  title={Share: Shared Low-Rank Subspaces for Efficient LoRA Adapter Management},
  year={2025},
  eprint={2602.06043},
  archivePrefix={arXiv},
}
```

## License

Apache 2.0
