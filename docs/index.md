# vlora

**Shared low-rank subspaces for efficient LoRA adapter management.**

Based on the [Share paper](https://arxiv.org/abs/2602.06043): LoRA adapters across tasks share a common low-rank subspace. Instead of storing *N* separate adapters, maintain **one shared basis** and **per-task coefficient vectors** — achieving up to 122× compression at scale.

## Install

```bash
pip install vlora-dev
```

## Quick Example

```python
from vlora import SharedSubspace, load_adapter

# Build shared subspace from existing adapters
adapters = [load_adapter(f"adapters/task_{i}") for i in range(5)]
subspace = SharedSubspace.from_adapters(adapters, num_components=16)

# Reconstruct any task back to full LoRA weights
weights = subspace.reconstruct("task_0")

# Add new adapters without rebuilding
new_adapter = load_adapter("adapters/new_task")
subspace.absorb_incremental(new_adapter, "new_task")
```

## Features

- **Compression**: Up to 122× storage reduction at scale
- **9 CLI commands**: compress, export, merge, analyze, validate, diff, benchmark, info, add
- **Adapter merging**: Task arithmetic, TIES, and DARE
- **Multi-task inference**: `VLoRAModel` for instant adapter switching
- **Training-in-subspace**: 100×+ parameter reduction
- **Task routing**: Lightweight MLP for per-input adapter blending
- **HuggingFace integration**: `VLoRACallback` for Trainer
- **Serving compatibility**: Export to vLLM, TGI, Ollama (via GGUF)

## Next Steps

- [Quickstart notebook](quickstart.md) — try vlora in Google Colab
- [Migration from PEFT](migration_from_peft.md) — integrate into your existing workflow
- [API Reference](api.md) — full documentation
