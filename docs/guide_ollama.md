# Ollama Integration Guide

Use vlora-compressed adapters with [Ollama](https://ollama.com/) for local inference.

## Overview

Ollama uses GGUF format for models and adapters. vlora exports PEFT-format adapters, which need to be converted to GGUF before use with Ollama.

## Workflow

### 1. Export from vlora

```bash
vlora export shared_subspace/ my_task -o exported_adapter/ \
  --alpha 32 --base-model meta-llama/Llama-3-8B \
  --target-modules q_proj,k_proj,v_proj,o_proj
```

### 2. Convert to GGUF

Use llama.cpp's conversion tool:

```bash
# Clone llama.cpp if you haven't
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert the PEFT adapter to GGUF
python convert_lora_to_gguf.py \
  --base meta-llama/Llama-3-8B \
  --lora-path ../exported_adapter/ \
  --outfile ../my_task.gguf
```

### 3. Create Ollama Modelfile

```dockerfile
# Modelfile
FROM llama3:8b
ADAPTER ./my_task.gguf
PARAMETER temperature 0.7
SYSTEM "You are a helpful assistant."
```

### 4. Run with Ollama

```bash
ollama create my-model -f Modelfile
ollama run my-model "Hello, how are you?"
```

## Batch Export Script

Export all tasks from a subspace:

```python
from vlora import SharedSubspace, save_adapter

subspace = SharedSubspace.load("shared_subspace/")

for task_id in subspace.tasks:
    weights = subspace.reconstruct(task_id)
    weights.metadata.update({
        "lora_alpha": 32,
        "base_model_name_or_path": "meta-llama/Llama-3-8B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    })
    save_adapter(weights, f"exported/{task_id}")
    print(f"Exported {task_id} — convert with convert_lora_to_gguf.py")
```

## Notes

- Ollama LoRA support requires Ollama ≥ 0.1.26
- GGUF conversion preserves the LoRA structure (A and B matrices)
- The base model in Ollama must match the model the adapters were trained on
- For fastest results, merge adapters before conversion:
  ```bash
  vlora merge adapters/task_0 adapters/task_1 -o merged/ --method ties
  ```
