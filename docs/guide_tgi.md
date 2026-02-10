# Text Generation Inference (TGI) Integration Guide

Serve vlora-compressed adapters with [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/).

## Export for TGI

TGI uses the same PEFT adapter format as vLLM:

```bash
vlora export shared_subspace/ my_task -o tgi_adapter/ \
  --alpha 32 \
  --base-model meta-llama/Llama-3-8B \
  --target-modules q_proj,k_proj,v_proj,o_proj
```

## Serve with TGI

### Docker

```bash
# Start TGI with LoRA support
docker run --gpus all \
  -v ./tgi_adapter:/adapters/my_task \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3-8B \
  --lora-adapters my_task=/adapters/my_task
```

### Python Client

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")

# Use the adapter
output = client.text_generation(
    "Summarize: ...",
    max_new_tokens=256,
    adapter_id="my_task",
)
```

## Multiple Adapters

Export all tasks and mount them:

```bash
for task in sentiment summarization translation; do
    vlora export shared_subspace/ $task -o tgi_adapters/$task/ \
      --alpha 32 --base-model meta-llama/Llama-3-8B \
      --target-modules q_proj,k_proj,v_proj,o_proj
done
```

```bash
docker run --gpus all \
  -v ./tgi_adapters:/adapters \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3-8B \
  --lora-adapters sentiment=/adapters/sentiment,summarization=/adapters/summarization,translation=/adapters/translation
```

## Dynamic Adapter Loading

For large numbers of adapters, export on-demand:

```python
from vlora import SharedSubspace, save_adapter

subspace = SharedSubspace.load("shared_subspace/")

def export_for_tgi(task_id: str, output_dir: str):
    weights = subspace.reconstruct(task_id)
    weights.metadata.update({
        "lora_alpha": 32,
        "base_model_name_or_path": "meta-llama/Llama-3-8B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    })
    save_adapter(weights, output_dir)
```
