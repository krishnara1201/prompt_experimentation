import pytest

from app.training.config import load_training_config
from app.training.dataset import DatasetBuildResult


@pytest.fixture
def fake_dataset(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"messages": []}\n')
    (tmp_path / "val.jsonl").write_text("")
    return DatasetBuildResult(
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        license_path=tmp_path / "LICENSE.txt",
        train_class_counts={"positive": 1},
        val_class_counts={},
        dropped_count=4,
        pool_size=1000,
        source="demo:5768",
    )


def test_prep_prints_stats(capsys, monkeypatch, fake_dataset):
    from scripts import finetune_prep

    monkeypatch.setattr(finetune_prep, "build_sft_dataset", lambda cfg: fake_dataset)
    finetune_prep.main(["--config", "training.yaml"])
    out = capsys.readouterr().out
    assert "dropped" in out.lower()
    assert "1000" in out


def test_train_dry_run_skips_training(capsys, monkeypatch, fake_dataset):
    from scripts import finetune_train

    monkeypatch.setattr(finetune_train, "build_sft_dataset", lambda cfg: fake_dataset)

    def _boom(*a, **k):
        raise AssertionError("run_training must not be called on --dry-run")

    monkeypatch.setattr(finetune_train, "run_training", _boom)
    finetune_train.main(["--config", "training.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert "dry run" in out.lower()


def test_train_calls_run_training(monkeypatch, fake_dataset):
    from scripts import finetune_train
    from app.training.train import TrainingResult

    calls = {}
    monkeypatch.setattr(finetune_train, "build_sft_dataset", lambda cfg: fake_dataset)
    monkeypatch.setattr(
        finetune_train,
        "run_training",
        lambda cfg, ds: calls.setdefault("r", TrainingResult(fake_dataset.train_path, [], 1.0, 42)),
    )
    finetune_train.main(["--config", "training.yaml"])
    assert "r" in calls


def test_export_prints_snippet(capsys, monkeypatch):
    from scripts import finetune_export
    from app.training.export import ExportResult

    monkeypatch.setattr(
        finetune_export,
        "export_arm",
        lambda cfg: ExportResult(
            gguf_path=cfg and None or None,
            modelfile_path=None,
            ollama_tag="ft-qwen3-8b",
            snippet_path=None,
            snippet="  - name: ft-qwen3-8b-local\n",
        ),
    )
    finetune_export.main(["--config", "training.yaml"])
    assert "ft-qwen3-8b-local" in capsys.readouterr().out
