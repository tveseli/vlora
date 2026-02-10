"""HuggingFace Trainer integration — VLoRACallback for training-in-subspace.

Usage with HuggingFace Trainer:
    from vlora import SharedSubspace, orthogonal_init
    from vlora.integrations.huggingface import VLoRACallback

    subspace = SharedSubspace.load("shared_subspace/")
    orthogonal_init(subspace, "new_task")

    callback = VLoRACallback(subspace, "new_task", lr=1e-3)
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[callback],
    )
    trainer.train()
    subspace.save("updated_subspace/")
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from vlora.subspace import SharedSubspace
from vlora.training import SubspaceTrainer

logger = logging.getLogger("vlora")

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _require_transformers():
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for HuggingFace integration. "
            "Install with: pip install vlora[hf]"
        )


if HAS_TRANSFORMERS:

    class VLoRACallback(TrainerCallback):
        """HuggingFace Trainer callback for training-in-subspace.

        Intercepts the training loop to optimize subspace loadings instead of
        full model parameters. Logs adapter-specific metrics (loadings norm,
        reconstruction error) to the Trainer's log history.

        Args:
            subspace: Shared subspace (task must already exist).
            task_id: Task whose loadings to train.
            lr: Learning rate for loadings optimizer.
            num_expand: Extra orthogonal directions for the optimizer.
            log_every: Log adapter metrics every N steps.
            save_on_end: Whether to call write_back() when training ends.
        """

        def __init__(
            self,
            subspace: SharedSubspace,
            task_id: str,
            lr: float = 1e-3,
            num_expand: int = 0,
            log_every: int = 50,
            save_on_end: bool = True,
        ):
            _require_transformers()
            self.subspace = subspace
            self.task_id = task_id
            self.lr = lr
            self.num_expand = num_expand
            self.log_every = log_every
            self.save_on_end = save_on_end
            self._trainer: SubspaceTrainer | None = None

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            self._trainer = SubspaceTrainer(
                self.subspace,
                self.task_id,
                lr=self.lr,
                num_expand=self.num_expand,
            )
            logger.info(
                "VLoRACallback: training '%s' with %d params (lr=%.1e)",
                self.task_id,
                self._trainer.num_trainable_params,
                self.lr,
            )

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            if self._trainer is None:
                return

            step = state.global_step
            if step > 0 and step % self.log_every == 0:
                # Log loadings norm as a proxy for adapter magnitude
                total_norm = 0.0
                for p in self._trainer.params.values():
                    total_norm += p.data.norm().item() ** 2
                total_norm = total_norm ** 0.5

                metrics = {
                    "vlora/loadings_norm": total_norm,
                    "vlora/trainable_params": self._trainer.num_trainable_params,
                    "vlora/step": self._trainer.step_count,
                }
                state.log_history.append(
                    {"step": step, **metrics}
                )
                logger.debug(
                    "VLoRACallback step %d: loadings_norm=%.4f",
                    step,
                    total_norm,
                )

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            if self._trainer is not None and self.save_on_end:
                self._trainer.write_back()
                logger.info(
                    "VLoRACallback: wrote back loadings for '%s' after %d steps",
                    self.task_id,
                    self._trainer.step_count,
                )

        @property
        def trainer(self) -> SubspaceTrainer | None:
            """Access the underlying SubspaceTrainer (available after on_train_begin)."""
            return self._trainer

else:
    # Stub class when transformers is not installed
    class VLoRACallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_transformers()
