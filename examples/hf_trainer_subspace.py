"""Example: Training-in-subspace with HuggingFace Trainer.

This shows how to use VLoRACallback to train only the subspace loadings
(a few hundred parameters) instead of full LoRA matrices (hundreds of thousands).

Usage:
    pip install vlora[hf]
    python examples/hf_trainer_subspace.py
"""

from vlora import SharedSubspace, load_adapter, orthogonal_init
from vlora.integrations.huggingface import VLoRACallback

# 1. Build or load a shared subspace
# subspace = SharedSubspace.load("shared_subspace/")

# For this example, create a synthetic one:
import torch
from vlora.io import LoRAWeights

layers = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"]
adapters = []
for i in range(5):
    lora_a = {l: torch.randn(16, 4096) for l in layers}
    lora_b = {l: torch.randn(4096, 16) for l in layers}
    adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=16))

subspace = SharedSubspace.from_adapters(adapters, num_components=8)
print(f"Subspace: {subspace.num_components} components, {len(subspace.layer_names)} layers")

# 2. Initialize a new task
orthogonal_init(subspace, "my_new_task")

# 3. Set up the callback
callback = VLoRACallback(
    subspace,
    task_id="my_new_task",
    lr=1e-3,
    log_every=10,
    save_on_end=True,  # auto write_back on train end
)

print(f"\nTo use with HuggingFace Trainer:")
print(f"  trainer = Trainer(")
print(f"      model=base_model,")
print(f"      args=training_args,")
print(f"      train_dataset=dataset,")
print(f"      callbacks=[callback],")
print(f"  )")
print(f"  trainer.train()")
print(f"  subspace.save('updated_subspace/')")
print(f"\nTrainable params: ~{callback._trainer is None and 'N/A (call on_train_begin first)' or 'initialized'}")
