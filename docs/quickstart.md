# Quickstart

Try vlora in Google Colab or locally.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tveseli/vlora/blob/main/examples/quickstart.ipynb)

## Install

```bash
pip install vlora
```

For HuggingFace Hub loading:
```bash
pip install vlora[hub]
```

For HuggingFace Trainer integration:
```bash
pip install vlora[hf]
```

## CLI Workflow

```bash
# Compress 5 adapters into a shared subspace
vlora compress adapters/task_* -o subspace/ -k 8

# Check compression stats
vlora info subspace/

# Export a task for serving
vlora export subspace/ task_0 -o exported/ --alpha 32

# Merge adapters
vlora merge adapters/task_0 adapters/task_1 -o merged/ --method ties

# Analyze similarity
vlora analyze adapters/task_*

# Health check
vlora validate subspace/
```

## Python Workflow

```python
from vlora import SharedSubspace, load_adapter, VLoRAModel

# Load and compress
adapters = [load_adapter(f"adapters/task_{i}") for i in range(5)]
subspace = SharedSubspace.from_adapters(adapters, num_components=8)

# Multi-task inference
model = VLoRAModel(base_model, subspace, lora_alpha=32)
model.set_task("task_0")
output = model(input_ids)

# Add new adapter incrementally
new_adapter = load_adapter("adapters/new_task")
subspace.absorb_incremental(new_adapter, "new_task")

# Merge adapters
from vlora import ties_merge
merged = ties_merge(adapters[:3], density=0.5)

# Save
subspace.save("subspace/")
```

See the [Colab notebook](https://colab.research.google.com/github/tveseli/vlora/blob/main/examples/quickstart.ipynb) for a complete interactive walkthrough.
