import textwrap

import pytest

from app.training.config import (
    InvalidTrainingConfigError,
    TrainingConfig,
    load_training_config,
)

VALID = textwrap.dedent(
    """
    run_name: demo
    base_model: unsloth/Qwen3-8B
    source_dataset: takala/financial_phrasebank
    source_config: sentences_75agree
    max_seq_len: 512
    lora_r: 16
    lora_alpha: 16
    lora_dropout: 0.0
    lora_target_modules: [q_proj, v_proj]
    epochs: 3
    learning_rate: 0.0002
    batch_size: 8
    grad_accum: 2
    seed: 42
    val_fraction: 0.1
    balance_neutral: false
    min_pool_size: 500
    gguf_quant: q4_k_m
    ollama_tag: ft-qwen3-8b
    output_dir: training/artifacts/demo
    """
)


def test_loads_valid_config(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID)
    cfg = load_training_config(str(p))
    assert isinstance(cfg, TrainingConfig)
    assert cfg.run_name == "demo"
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.epochs == 3.0
    assert cfg.balance_neutral is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID + "\nmystery_key: 1\n")
    with pytest.raises(InvalidTrainingConfigError, match="mystery_key"):
        load_training_config(str(p))


def test_missing_required_key_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID.replace("run_name: demo", ""))
    with pytest.raises(InvalidTrainingConfigError, match="run_name"):
        load_training_config(str(p))


def test_wrong_type_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID.replace("max_seq_len: 512", "max_seq_len: not-an-int"))
    with pytest.raises(InvalidTrainingConfigError, match="max_seq_len"):
        load_training_config(str(p))


def test_committed_training_yaml_parses():
    cfg = load_training_config("training.yaml")
    assert cfg.ollama_tag
    assert cfg.gguf_quant == "q4_k_m"
