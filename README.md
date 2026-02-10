<p align="center">
  <img src="logo.png" alt="vLoRA" width="400">
</p>

<p align="center">
  <strong>Shared low-rank subspaces for efficient LoRA adapter management.</strong>
</p>

Based on the [Share paper](https://arxiv.org/abs/2602.06043): LoRA adapters across tasks share a common low-rank subspace. Instead of storing *N* separate adapters, maintain **one shared basis** and **per-task coefficient vectors** — achieving up to 122× compression at scale.

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

## CLI

vlora ships with a command-line tool for common workflows:

```bash
# Build a shared subspace from adapter directories
vlora compress adapters/task_0 adapters/task_1 adapters/task_2 -o shared_subspace/

# Inspect a subspace
vlora info shared_subspace/

# Export a task back to PEFT format
vlora export shared_subspace/ task_0 -o exported_adapter/

# Add a new adapter to an existing subspace
vlora add shared_subspace/ adapters/new_task --task-id new_task

# Analyze adapter similarity and clustering
vlora analyze adapters/task_0 adapters/task_1 adapters/task_2
```

## Multi-Task Inference

Wrap any PyTorch model with `VLoRAModel` for on-the-fly adapter switching:

```python
from vlora import VLoRAModel, SharedSubspace

subspace = SharedSubspace.load("shared_subspace/")
model = VLoRAModel(base_model, subspace)

# Switch adapters instantly — reconstructed from compressed loadings
model.set_task("task_0")
output = model(input_ids)

model.set_task("task_1")  # cached if same task
output = model(input_ids)

print(model.available_tasks)  # ["task_0", "task_1", ...]
```

## Training in the Subspace

Train only the loadings vector (k params per layer) instead of full LoRA matrices — 100×+ parameter reduction:

```python
from vlora import SharedSubspace, orthogonal_init, SubspaceTrainer

subspace = SharedSubspace.load("shared_subspace/")
orthogonal_init(subspace, "new_task")  # initialize near-zero

trainer = SubspaceTrainer(subspace, "new_task", lr=1e-3)
print(f"Trainable params: {trainer.num_trainable_params}")  # e.g. 192 vs 200K

for batch in dataloader:
    loss = compute_loss(model, batch)
    trainer.step(loss)

trainer.write_back()  # persist learned loadings
subspace.save("updated_subspace/")
```

## Task Router

Automatically blend adapters per input using a lightweight router:

```python
from vlora import TaskRouter, SharedSubspace

subspace = SharedSubspace.load("shared_subspace/")
router = TaskRouter.from_subspace(subspace, input_dim=4096)

# Router produces soft blend weights over tasks
x = get_input_embedding(batch)  # (B, 4096)
blended = router.blend_loadings(x, subspace)
subspace.tasks["__routed__"] = blended
recon = subspace.reconstruct("__routed__")
```

## Adapter Analysis

Analyze relationships between adapters before compression:

```python
from vlora import load_adapter, compute_similarity_matrix, find_clusters, adapter_diff

adapters = [load_adapter(f"adapters/task_{i}") for i in range(10)]

# Pairwise cosine similarity
sim_matrix = compute_similarity_matrix(adapters)

# Find redundant adapter groups
clusters = find_clusters(sim_matrix, threshold=0.9)

# Per-layer comparison of two adapters
diff = adapter_diff(adapters[0], adapters[1])
```

## Advanced Compression

```python
# Adaptive k: different components per layer based on explained variance
subspace = SharedSubspace.from_adapters(adapters, adaptive_k=True, variance_threshold=0.9)

# Quantize components for smaller memory footprint
subspace.quantize(bits=8)  # or bits=4

# Check compression stats
stats = subspace.compression_stats()
print(f"Compression ratio: {stats['compression_ratio']:.1f}×")
print(f"Compressed: {stats['total_params_compressed']:,} params")
print(f"Original:   {stats['total_params_original']:,} params")
```

## Incremental Updates

Scale to thousands of adapters without loading them all at once:

```python
# Streaming: load adapters one at a time from disk
subspace = SharedSubspace.from_adapters_streaming(
    adapter_paths, num_components=8
)

# Incremental absorb: fast O(1) update without full SVD recompute
subspace.absorb_incremental(new_adapter, "new_task")

# Move to GPU / change precision
subspace.to(device="cuda", dtype=torch.float16)
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
  - `.from_adapters_streaming(paths, ...)` — Build one adapter at a time from disk
  - `.project(adapter, task_id)` → `TaskProjection`
  - `.add_task(projection)` — Register a projected task
  - `.reconstruct(task_id)` → `LoRAWeights`
  - `.absorb(adapter, task_id)` — Incorporate + recompute (full SVD)
  - `.absorb_incremental(adapter, task_id)` — Fast incremental update
  - `.get_trainable_params(task_id)` — For training integration
  - `.quantize(bits=8)` — Quantize components (int8/int4)
  - `.compression_stats()` — Compression ratio and parameter counts
  - `.to(device, dtype)` — Move tensors to device/dtype
  - `.save(path)` / `.load(path)` — Serialization

### Model Integration

- **`VLoRAModel(base_model, subspace)`** — Inference wrapper with forward hooks
  - `.set_task(task_id)` — Switch adapter (cached)
  - `.clear_task()` — Remove adapter
  - `.available_tasks` — List task IDs
  - `.reconstruct_state_dict(task_id)` — Get delta weight dict

### Training

- **`orthogonal_init(subspace, task_id)`** — Initialize new task with small loadings
- **`SubspaceTrainer(subspace, task_id)`** — Optimizer wrapper for loadings-only training
  - `.step(loss)` — Backprop + update
  - `.write_back()` — Persist to subspace

### Router

- **`TaskRouter(input_dim, num_tasks)`** — Lightweight adapter routing MLP
  - `.from_subspace(subspace, input_dim)` — Auto-create from subspace
  - `.blend_loadings(x, subspace)` — Per-input adapter blending

### Analysis

- **`compute_similarity_matrix(adapters)`** — Pairwise cosine similarity
- **`find_clusters(sim_matrix, threshold)`** — Greedy clustering
- **`adapter_diff(a, b)`** — Per-layer L2 distance + cosine similarity
- **`subspace_coverage(subspace, adapter)`** — How well subspace represents an adapter

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
- `incremental_svd_update`

## Benchmarks — Real-World Adapters

Tested with 8 [Lots-of-LoRAs](https://huggingface.co/Lots-of-LoRAs) adapters (Mistral-7B, rank 16, 96 layers each):

**Variance explained** — the B matrices share structure much more strongly:

| k | Variance (A) | Variance (B) |
|---|-------------|-------------|
| 1 | 0.19 | 0.43 |
| 2 | 0.37 | 0.73 |
| 4 | 0.69 | 0.95 |
| 6 | 1.00 | 1.00 |

**Reconstruction error** (relative L2 norm):

| k | Mean Error | Max Error |
|---|-----------|-----------|
| 1 | 0.826 | 0.938 |
| 4 | 0.387 | 0.846 |
| 6 | 0.000002 | 0.000003 |

**Compression at scale** — shared basis is a one-time cost; each new adapter adds only k loadings per layer:

| N adapters | Full (MB) | vLoRA (MB) | Ratio |
|-----------|----------|-----------|-------|
| 8 | 288 | 288 | 1.0× |
| 100 | 3,600 | 289 | 12.5× |
| 1,000 | 36,000 | 293 | 122.8× |

Run the benchmark yourself:
```bash
pip install vlora[hub]
python examples/real_adapters.py
```

## Dependencies

- `torch >= 2.0`
- `safetensors >= 0.4`
- `click >= 8.0`
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
