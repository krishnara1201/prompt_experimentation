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


def test_train_reuse_dataset_skips_build(tmp_path, monkeypatch):
    from dataclasses import replace

    from scripts import finetune_train
    from app.training.config import load_training_config
    from app.training.train import TrainingResult

    (tmp_path / "train.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "positive"}]}\n'
        '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "positive"}]}\n'
    )
    (tmp_path / "val.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "neutral"}]}\n'
    )
    base = load_training_config("training.yaml")
    monkeypatch.setattr(
        finetune_train, "load_training_config", lambda _p: replace(base, output_dir=str(tmp_path))
    )

    def _boom(cfg):
        raise AssertionError("build_sft_dataset must not run with --reuse-dataset")

    monkeypatch.setattr(finetune_train, "build_sft_dataset", _boom)

    captured = {}

    def _fake_train(cfg, ds):
        captured["ds"] = ds
        return TrainingResult(tmp_path / "adapter", [], 1.0, 42)

    monkeypatch.setattr(finetune_train, "run_training", _fake_train)

    finetune_train.main(["--config", "x", "--reuse-dataset"])
    ds = captured["ds"]
    assert ds.source == "(reused)"
    assert ds.pool_size == 3
    assert ds.dropped_count == -1
    assert ds.train_class_counts == {"positive": 2}
    assert ds.val_class_counts == {"neutral": 1}


def test_train_reuse_dataset_missing_files_errors(tmp_path, monkeypatch):
    from dataclasses import replace

    from scripts import finetune_train
    from app.training.config import load_training_config

    base = load_training_config("training.yaml")
    monkeypatch.setattr(
        finetune_train, "load_training_config", lambda _p: replace(base, output_dir=str(tmp_path))
    )
    with pytest.raises(SystemExit):
        finetune_train.main(["--config", "x", "--reuse-dataset"])


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
            gguf_path=None,
            modelfile_path=None,
            ollama_tag="ft-qwen3-8b",
            snippet_path=None,
            snippet="  - name: ft-qwen3-8b-local\n",
        ),
    )
    finetune_export.main(["--config", "training.yaml"])
    assert "ft-qwen3-8b-local" in capsys.readouterr().out
