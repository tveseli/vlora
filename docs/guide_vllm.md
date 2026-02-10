# vLLM Integration Guide

Serve vlora-compressed adapters with [vLLM](https://docs.vllm.ai/) for high-throughput multi-adapter inference.

## Export for vLLM

vLLM expects PEFT-format adapters. Use `vlora export` with the appropriate metadata:

```bash
vlora export shared_subspace/ my_task -o vllm_adapter/ \
  --alpha 32 \
  --base-model meta-llama/Llama-3-8B \
  --target-modules q_proj,k_proj,v_proj,o_proj
```

This creates:
```
vllm_adapter/
├── adapter_model.safetensors
└── adapter_config.json
```

The `adapter_config.json` includes all fields vLLM needs:
```json
{
  "r": 16,
  "lora_alpha": 32,
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "bias": "none",
  "lora_dropout": 0.0,
  "base_model_name_or_path": "meta-llama/Llama-3-8B",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

## Serve with vLLM

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3-8B",
    enable_lora=True,
    max_lora_rank=16,
)

# Serve the exported adapter
output = llm.generate(
    "Summarize the following article:",
    sampling_params=SamplingParams(max_tokens=256),
    lora_request=LoRARequest("my_task", 1, "vllm_adapter/"),
)
```

## Multi-Adapter Serving

Export multiple tasks and serve them concurrently:

```bash
for task in sentiment summarization translation; do
    vlora export shared_subspace/ $task -o vllm_adapters/$task/ \
      --alpha 32 --base-model meta-llama/Llama-3-8B \
      --target-modules q_proj,k_proj,v_proj,o_proj
done
```

```python
# vLLM handles concurrent adapter requests
requests = [
    LoRARequest("sentiment", 1, "vllm_adapters/sentiment/"),
    LoRARequest("summarization", 2, "vllm_adapters/summarization/"),
    LoRARequest("translation", 3, "vllm_adapters/translation/"),
]
```

## Batch Export Script

```python
from vlora import SharedSubspace, save_adapter

subspace = SharedSubspace.load("shared_subspace/")

for task_id in subspace.tasks:
    weights = subspace.reconstruct(task_id)
    weights.metadata["lora_alpha"] = 32
    weights.metadata["base_model_name_or_path"] = "meta-llama/Llama-3-8B"
    weights.metadata["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    save_adapter(weights, f"vllm_adapters/{task_id}")
    print(f"Exported {task_id}")
```
