# Migrating from PEFT to vlora

This guide shows how to integrate vlora into an existing PEFT/LoRA workflow. vlora reads standard PEFT adapter format (safetensors + adapter_config.json) — no changes to your training pipeline are needed.

## What Changes, What Doesn't

| Aspect | Before (PEFT only) | After (PEFT + vlora) |
|--------|-------------------|---------------------|
| Training | Train LoRA adapters normally | Same — no changes needed |
| Storage | N × full adapter files | 1 shared basis + N × tiny loadings |
| Loading | Load each adapter separately | Reconstruct from subspace |
| Serving | One adapter per model instance | Switch adapters instantly |
| Format | PEFT safetensors | PEFT safetensors (compatible) |

## Step 1: Compress Existing Adapters

You already have PEFT adapters on disk:

```
adapters/
├── sentiment/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── summarization/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
└── translation/
    ├── adapter_model.safetensors
    └── adapter_config.json
```

Compress them into a shared subspace:

```bash
vlora compress adapters/sentiment adapters/summarization adapters/translation \
  -o shared_subspace/ -k 4
```

Or in Python:

```python
from vlora import load_adapter, SharedSubspace

adapters = [
    load_adapter("adapters/sentiment"),
    load_adapter("adapters/summarization"),
    load_adapter("adapters/translation"),
]
subspace = SharedSubspace.from_adapters(
    adapters,
    task_ids=["sentiment", "summarization", "translation"],
    num_components=4,
)
subspace.save("shared_subspace/")
```

## Step 2: Use the Subspace

### Reconstruct for Serving

Export any task back to standard PEFT format for vLLM/TGI:

```bash
vlora export shared_subspace/ sentiment -o exported/sentiment/ \
  --alpha 32 --base-model meta-llama/Llama-3-8B
```

### Switch Adapters at Inference

```python
from vlora import VLoRAModel, SharedSubspace

subspace = SharedSubspace.load("shared_subspace/")
model = VLoRAModel(base_model, subspace, lora_alpha=32)

model.set_task("sentiment")
output = model(input_ids)

model.set_task("translation")  # instant switch
output = model(input_ids)
```

### Add New Adapters Later

When you train a new adapter, add it without rebuilding:

```bash
vlora add shared_subspace/ adapters/new_task --task-id new_task --incremental
```

## Step 3: Train New Tasks Efficiently

Instead of training full LoRA (rank × dim params per layer), train only subspace loadings (k params per layer):

```python
from vlora import orthogonal_init, SubspaceTrainer

subspace = SharedSubspace.load("shared_subspace/")
orthogonal_init(subspace, "new_task")

trainer = SubspaceTrainer(subspace, "new_task", lr=1e-3)
# trainer.num_trainable_params → ~192 instead of ~200,000

for batch in dataloader:
    loss = compute_loss(model, batch)
    trainer.step(loss)

trainer.write_back()
subspace.save("shared_subspace/")
```

## FAQ

**Q: Do I need to change my training pipeline?**
No. Train LoRA adapters with PEFT/Axolotl/unsloth as usual. vlora operates on the saved adapter files.

**Q: Can I still load exported adapters with PEFT?**
Yes. `vlora export` produces standard PEFT-format files (adapter_model.safetensors + adapter_config.json).

**Q: What LoRA ranks are supported?**
Any rank. All adapters in a subspace must have the same rank.

**Q: How many components (k) should I use?**
Start with `k=4` for a good balance. Use `--adaptive-k` for automatic per-layer selection. Check variance explained with `vlora info`.

**Q: Does vlora support QLoRA / quantized base models?**
vlora operates on the LoRA adapter weights only — it doesn't touch the base model. QLoRA adapters work fine.
