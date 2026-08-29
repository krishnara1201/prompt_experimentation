"""Turns a trained LoRA adapter into a running Ollama model and the
arms.yaml entry that points at it. Host-side, GPU-only for the merge/GGUF
steps.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.training.config import TrainingConfig
from app.training.train import import_unsloth
# Re-exported so tests (and callers) can reference export.MissingTrainingDepsError.
from app.training.train import MissingTrainingDepsError  # noqa: F401

DEFAULT_ARMS_PATH = Path(__file__).resolve().parent.parent.parent / "arms.yaml"
FT_ARM_NAME = "ft-qwen3-8b-local"


class ExportError(Exception):
    pass


@dataclass(frozen=True)
class ExportResult:
    gguf_path: Path
    modelfile_path: Path
    ollama_tag: str
    snippet_path: Path
    snippet: str


def build_modelfile(cfg: TrainingConfig, gguf_filename: str) -> str:
    # Qwen3 honours a `/no_think` directive in the system prompt: it
    # suppresses the <think> block so the arm emits a bare label and the
    # shared eval prompt (app/eval_prompt.py) is passed through untouched.
    return "\n".join(
        [
            f"FROM ./{gguf_filename}",
            'SYSTEM """/no_think"""',
            # max_seq_len is the *training* sequence length; num_ctx must also
            # cover the completion the arm is asked for (render_arm_snippet
            # sets max_tokens: 1024), so give it that much headroom.
            f"PARAMETER num_ctx {cfg.max_seq_len + 1024}",
            "PARAMETER temperature 0",
            'PARAMETER stop "<|im_end|>"',
            "",
        ]
    )


def read_local_arm_base_url(arms_path: Path, arm_name: str = "qwen3-8b-local") -> str:
    raw = yaml.safe_load(Path(arms_path).read_text())
    for entry in raw.get("arms", []):
        if entry.get("name") == arm_name:
            url = entry.get("base_url")
            if not url:
                raise ExportError(f"arm '{arm_name}' in {arms_path} has no base_url")
            return url
    raise ExportError(f"arm '{arm_name}' not found in {arms_path}")


def render_arm_snippet(cfg: TrainingConfig, base_url: str) -> str:
    return "\n".join(
        [
            f"  - name: {FT_ARM_NAME}",
            "    adapter: openai_compatible",
            f"    base_url: {base_url}",
            f"    model: {cfg.ollama_tag}",
            "    max_tokens: 1024",
            "",
        ]
    )


def export_arm(cfg: TrainingConfig, arms_path: Path | None = None) -> ExportResult:
    import_unsloth()
    from unsloth import FastLanguageModel

    out_dir = Path(cfg.output_dir)
    adapter_path = out_dir / "adapter"
    if not adapter_path.exists():
        raise ExportError(f"no trained adapter at {adapter_path} -- run `pe finetune train` first")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=False,
        dtype=None,
    )

    # Save the merged 16-bit HF model first: it is the documented
    # manual-recovery path (backend/README.md 'Fallbacks' ->
    # training/artifacts/<run_name>/merged/) if the GGUF toolchain step fails.
    model.save_pretrained_merged(
        str(out_dir / "merged"), tokenizer, save_method="merged_16bit"
    )

    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_dir), tokenizer, quantization_method=cfg.gguf_quant
    )
    gguf_files = sorted(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise ExportError(f"GGUF conversion produced no .gguf file in {gguf_dir}")
    gguf_path = gguf_files[0]

    modelfile_path = gguf_dir / "Modelfile"
    modelfile_path.write_text(build_modelfile(cfg, gguf_path.name), encoding="utf-8")

    proc = subprocess.run(
        ["ollama", "create", cfg.ollama_tag, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ExportError(f"`ollama create` failed:\n{proc.stderr}")

    base_url = read_local_arm_base_url(arms_path or DEFAULT_ARMS_PATH)
    snippet = render_arm_snippet(cfg, base_url)
    snippet_path = out_dir / "arm_snippet.yaml"
    snippet_path.write_text(snippet, encoding="utf-8")

    return ExportResult(
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
        ollama_tag=cfg.ollama_tag,
        snippet_path=snippet_path,
        snippet=snippet,
    )
