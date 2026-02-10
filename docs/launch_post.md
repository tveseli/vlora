# Launch Post — r/LocalLLaMA + r/MachineLearning

## Title

**vLoRA: 122× compression for LoRA adapters — shared low-rank subspaces with CLI, merging, and multi-task inference**

## Body

We've been working on **vlora**, an open-source library for managing collections of LoRA adapters efficiently using shared low-rank subspaces.

### The Problem

If you're running multiple fine-tuned models (sentiment, summarization, translation, etc.), you're storing N copies of almost-identical LoRA weight matrices. For 1000 adapters on Mistral-7B rank-16, that's **36 GB** of adapter weights — most of which is redundant.

### The Insight

LoRA adapters trained on related tasks share a common low-rank subspace ([arXiv:2602.06043](https://arxiv.org/abs/2602.06043)). vlora discovers this shared basis via SVD and represents each adapter as a tiny coefficient vector instead of full weight matrices.

### Results

| N adapters | Full Storage | vLoRA | Ratio |
|-----------|-------------|-------|-------|
| 8 | 288 MB | 288 MB | 1.0× |
| 100 | 3.6 GB | 289 MB | 12.5× |
| 1,000 | 36 GB | 293 MB | **122.8×** |

At k=4 components, B-side matrices explain 95% of variance across 8 Mistral-7B adapters.

### What's in the box

```bash
pip install vlora-dev
```

**CLI** (9 commands):
```bash
vlora compress adapters/* -o subspace/ -k 8
vlora merge adapters/* -o merged/ --method ties
vlora export subspace/ task_0 -o exported/ --alpha 32
vlora info subspace/ --json
vlora analyze adapters/*
vlora validate subspace/
```

**Python API**:
- `SharedSubspace.from_adapters()` → build shared basis
- `VLoRAModel(base_model, subspace)` → instant adapter switching
- `SubspaceTrainer` → train only k loadings per layer (100×+ param reduction)
- `TaskRouter` → per-input adapter blending
- `task_arithmetic`, `ties_merge`, `dare_merge` → adapter merging
- `find_outliers`, `compute_similarity_matrix` → adapter analysis

**Integrations**:
- HuggingFace Trainer callback (`VLoRACallback`)
- Export to vLLM, TGI, Ollama (PEFT-compatible format)
- Incremental updates — add adapters without full recompute

**Docs**: Migration guide, serving guides, Colab notebook, MkDocs API reference.

### Links

- GitHub: https://github.com/tveseli/vlora
- PyPI: https://pypi.org/project/vlora-dev/
- Paper: https://arxiv.org/abs/2602.06043
- Website: https://vlora-dev.web.app

Apache 2.0. PyTorch ≥ 2.0.

Happy to answer questions about the approach or help with integration!
