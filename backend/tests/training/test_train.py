import pytest

from app.training import train


def test_import_unsloth_raises_actionable_error_when_absent():
    try:
        import unsloth  # noqa: F401
    except ImportError:
        with pytest.raises(train.MissingTrainingDepsError, match="uv sync --extra training"):
            train.import_unsloth()
    else:
        pytest.skip("unsloth is installed in this environment")


def test_run_training_raises_without_deps(tmp_path):
    try:
        import unsloth  # noqa: F401
    except ImportError:
        from app.training.dataset import DatasetBuildResult

        fake = DatasetBuildResult(
            train_path=tmp_path / "train.jsonl",
            val_path=tmp_path / "val.jsonl",
            license_path=tmp_path / "LICENSE.txt",
            train_class_counts={},
            val_class_counts={},
            dropped_count=0,
            pool_size=0,
            source="x",
        )
        from app.training.config import load_training_config

        with pytest.raises(train.MissingTrainingDepsError):
            train.run_training(load_training_config("training.yaml"), fake)
    else:
        pytest.skip("unsloth is installed in this environment")
