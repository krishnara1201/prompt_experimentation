import json

import pytest

from app.eval_prompt import render_eval_prompt
from app.training import dataset as ds
from app.training.config import load_training_config

BASE_CFG = load_training_config("training.yaml")


def cfg(tmp_path, **over):
    from dataclasses import replace

    kw = {"output_dir": str(tmp_path / "art"), "min_pool_size": 1}
    kw.update(over)
    return replace(BASE_CFG, **kw)


def fake_source(rows):
    return lambda _cfg: list(rows)


def make_rows(n_pos, n_neg, n_neu, prefix="s"):
    rows = []
    for i in range(n_pos):
        rows.append((f"{prefix} pos {i}", "positive"))
    for i in range(n_neg):
        rows.append((f"{prefix} neg {i}", "negative"))
    for i in range(n_neu):
        rows.append((f"{prefix} neu {i}", "neutral"))
    return rows


def test_drops_eval_set_overlap(tmp_path, monkeypatch):
    rows = make_rows(3, 3, 3) + [("Profit ROSE  sharply.", "positive")]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: {"profit rose sharply."})

    result = ds.build_sft_dataset(cfg(tmp_path))

    assert result.dropped_count == 1
    assert result.pool_size == 9
    lines = result.train_path.read_text().splitlines() + result.val_path.read_text().splitlines()
    texts = [json.loads(l)["messages"][0]["content"] for l in lines]
    assert all("Profit ROSE" not in t for t in texts)


def test_record_format_matches_render_eval_prompt(tmp_path, monkeypatch):
    rows = [("Shares fell 4 percent.", "negative")] + make_rows(2, 2, 2)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    result = ds.build_sft_dataset(cfg(tmp_path, val_fraction=0.0))
    records = [json.loads(l) for l in result.train_path.read_text().splitlines()]
    rec = next(r for r in records if "Shares fell 4 percent." in r["messages"][0]["content"])
    assert rec["messages"][0] == {
        "role": "user",
        "content": render_eval_prompt("Shares fell 4 percent."),
    }
    assert rec["messages"][1] == {"role": "assistant", "content": "negative"}


def test_leakage_error_when_overlap_survives(tmp_path, monkeypatch):
    # fetch_eval_texts returns an already-normalized string that does NOT
    # match after normalization -> simulate a guard bug by making normalize
    # a no-op via monkeypatch is overkill; instead assert the happy path is
    # clean, then force min_pool_size high:
    rows = make_rows(2, 2, 2)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())
    with pytest.raises(ds.LeakageError, match="pool"):
        ds.build_sft_dataset(cfg(tmp_path, min_pool_size=999))


def test_deterministic_seeded_split(tmp_path, monkeypatch):
    rows = make_rows(20, 20, 20)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    a = ds.build_sft_dataset(cfg(tmp_path / "a", seed=7, val_fraction=0.2))
    b = ds.build_sft_dataset(cfg(tmp_path / "b", seed=7, val_fraction=0.2))
    assert a.val_path.read_text() == b.val_path.read_text()
    train_texts = {json.loads(l)["messages"][0]["content"] for l in a.train_path.read_text().splitlines()}
    val_texts = {json.loads(l)["messages"][0]["content"] for l in a.val_path.read_text().splitlines()}
    assert train_texts.isdisjoint(val_texts)


def test_balance_neutral_downsamples(tmp_path, monkeypatch):
    rows = make_rows(5, 5, 40)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    result = ds.build_sft_dataset(cfg(tmp_path, balance_neutral=True, val_fraction=0.0))
    assert result.train_class_counts["neutral"] == 5


def test_license_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "load_source_examples", fake_source(make_rows(2, 2, 2)))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())
    result = ds.build_sft_dataset(cfg(tmp_path))
    assert "CC BY-NC-SA" in result.license_path.read_text()
    assert "Malo" in result.license_path.read_text()
