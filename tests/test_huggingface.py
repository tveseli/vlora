"""Tests for vlora.integrations.huggingface — HF Trainer callback."""

import torch
import pytest

from vlora.io import LoRAWeights
from vlora.subspace import SharedSubspace
from vlora.training import orthogonal_init


def _make_subspace():
    """Create a small subspace for testing."""
    layers = ["layer.0.q_proj", "layer.0.v_proj"]
    shared_a = {l: torch.randn(3, 4 * 32) for l in layers}
    shared_b = {l: torch.randn(3, 32 * 4) for l in layers}

    adapters = []
    for i in range(3):
        lora_a = {l: (torch.randn(3) @ shared_a[l]).reshape(4, 32) for l in layers}
        lora_b = {l: (torch.randn(3) @ shared_b[l]).reshape(32, 4) for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=4))

    return SharedSubspace.from_adapters(adapters, num_components=2)


class TestVLoRACallbackImport:
    def test_import_without_transformers(self):
        """VLoRACallback should be importable even without transformers."""
        from vlora.integrations.huggingface import VLoRACallback
        assert VLoRACallback is not None

    def test_stub_raises_without_transformers(self):
        """If transformers not installed, instantiation raises ImportError."""
        try:
            import transformers
            pytest.skip("transformers is installed")
        except ImportError:
            from vlora.integrations.huggingface import VLoRACallback
            with pytest.raises(ImportError, match="transformers"):
                VLoRACallback(None, "test")


def _can_use_training_args():
    """Check if TrainingArguments can be instantiated (needs accelerate)."""
    try:
        from transformers import TrainingArguments
        TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        return True
    except (ImportError, Exception):
        return False


class TestVLoRACallbackWithTransformers:
    @pytest.fixture(autouse=True)
    def skip_without_full_hf(self):
        if not _can_use_training_args():
            pytest.skip("transformers + accelerate not installed")

    def test_callback_creates_trainer_on_begin(self):
        from vlora.integrations.huggingface import VLoRACallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3)
        assert callback.trainer is None

        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        control = TrainerControl()
        callback.on_train_begin(args, state, control)

        assert callback.trainer is not None
        assert callback.trainer.num_trainable_params > 0

    def test_callback_write_back_on_end(self):
        from vlora.integrations.huggingface import VLoRACallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3, save_on_end=True)
        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        control = TrainerControl()

        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)

        assert "test_task" in sub.tasks

    def test_callback_logs_metrics(self):
        from vlora.integrations.huggingface import VLoRACallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3, log_every=1)
        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        state.global_step = 1
        control = TrainerControl()

        callback.on_train_begin(args, state, control)
        callback.on_step_end(args, state, control)

        vlora_logs = [l for l in state.log_history if "vlora/loadings_norm" in l]
        assert len(vlora_logs) == 1
        assert vlora_logs[0]["vlora/loadings_norm"] >= 0
