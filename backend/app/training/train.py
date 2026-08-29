"""QLoRA fine-tune of the local model with Unsloth. Host-side, GPU-only.

Saves the LoRA adapter only -- merge / GGUF / ollama create is export.py,
so a training run can be inspected before it becomes an Ollama model.
Training is seeded but not bit-reproducible.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from app.training.config import TrainingConfig
from app.training.dataset import DatasetBuildResult

_INSTALL_HINT = (
    "Training dependencies are not installed. Run `uv sync --extra training` "
    "on a machine with a CUDA GPU (see backend/README.md 'Phase 7')."
)


class MissingTrainingDepsError(RuntimeError):
    pass


def import_unsloth() -> ModuleType:
    # Guard the whole training stack, not just unsloth: a partial install
    # (unsloth present, trl/datasets/peft/transformers missing) would
    # otherwise surface as a raw ImportError deep in run_training/export_arm.
    # unsloth is imported first -- it patches transformers on import.
    try:
        import unsloth
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise MissingTrainingDepsError(_INSTALL_HINT) from exc
    return unsloth


@dataclass(frozen=True)
class TrainingResult:
    adapter_path: Path
    loss_history: list[dict]
    wall_seconds: float
    seed: int


def _load_chat_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_training(cfg: TrainingConfig, dataset: DatasetBuildResult) -> TrainingResult:
    import_unsloth()  # fail fast with the actionable message
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    started = time.perf_counter()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    def _to_text(records: list[dict]) -> Dataset:
        rows = []
        for rec in records:
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            rows.append({"text": text})
        return Dataset.from_list(rows)

    train_ds = _to_text(_load_chat_records(dataset.train_path))
    eval_ds = _to_text(_load_chat_records(dataset.val_path)) if dataset.val_path.exists() else None

    out_dir = Path(cfg.output_dir)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            logging_steps=1,
            eval_strategy="epoch" if eval_ds is not None else "no",
            output_dir=str(out_dir / "trainer"),
            report_to=[],
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_len,
        ),
    )
    # Train only on the assistant tokens (the label word), not the prompt.
    # The instruction_part / response_part markers and enable_thinking=False
    # are Qwen3-specific and are on the Task 10 verify list -- a mismatch shows
    # up as the model learning to echo the prompt.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    loss_history = [
        {k: v for k, v in entry.items() if k in ("epoch", "step", "loss", "eval_loss")}
        for entry in trainer.state.log_history
    ]
    return TrainingResult(
        adapter_path=adapter_path,
        loss_history=loss_history,
        wall_seconds=time.perf_counter() - started,
        seed=cfg.seed,
    )
