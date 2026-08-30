from dataclasses import replace
from pathlib import Path

import pytest

from app.training import export
from app.training.config import load_training_config

CFG = load_training_config("training.yaml")
FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "arms_sample.yaml"


def test_build_modelfile_disables_thinking_and_pins_params():
    text = export.build_modelfile(CFG, "qwen3-8b-finsent-lora.Q4_K_M.gguf")
    assert "FROM ./qwen3-8b-finsent-lora.Q4_K_M.gguf" in text
    # num_ctx must leave completion headroom over the training seq len.
    assert f"num_ctx {CFG.max_seq_len + 1024}" in text
    assert "temperature 0" in text
    assert "/no_think" in text  # thinking disabled for classification


def test_read_local_arm_base_url():
    assert export.read_local_arm_base_url(FIXTURE) == "http://172.30.0.127:11434/v1"


def test_read_local_arm_base_url_missing_arm():
    with pytest.raises(export.ExportError, match="ghost"):
        export.read_local_arm_base_url(FIXTURE, arm_name="ghost")


def test_render_arm_snippet():
    snippet = export.render_arm_snippet(CFG, "http://172.30.0.127:11434/v1")
    assert "name: ft-qwen3-8b-local" in snippet
    assert "adapter: openai_compatible" in snippet
    assert "model: ft-qwen3-8b" in snippet
    assert "base_url: http://172.30.0.127:11434/v1" in snippet


def test_export_arm_raises_without_deps(tmp_path):
    # A trained adapter exists but no GGUF yet, so export_arm falls through to
    # the merge/convert path -- which needs unsloth.
    (tmp_path / "adapter").mkdir()
    try:
        import unsloth  # noqa: F401
    except ImportError:
        with pytest.raises(export.MissingTrainingDepsError):
            export.export_arm(replace(CFG, output_dir=str(tmp_path)), arms_path=FIXTURE)
    else:
        pytest.skip("unsloth is installed in this environment")


def test_export_arm_raises_without_trained_adapter(tmp_path):
    with pytest.raises(export.ExportError, match="no trained adapter"):
        export.export_arm(replace(CFG, output_dir=str(tmp_path)), arms_path=FIXTURE)
