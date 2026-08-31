import json

import pytest

from app.config.tasks import load_task
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


# A sentinel training row that overlaps the eval set, so the leakage guard
# sees a non-empty eval set AND a non-zero drop (both now required -- the
# guard fails closed otherwise, see build_sft_dataset).
LEAK_ROW = ("leaked eval sentence", "neutral")
LEAK_EVAL = {"leaked eval sentence"}


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
    sentence = "Shares fell 4 percent."
    rows = [(sentence, "negative")] + make_rows(2, 2, 2) + [LEAK_ROW]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)

    result = ds.build_sft_dataset(cfg(tmp_path, val_fraction=0.0))
    records = [json.loads(l) for l in result.train_path.read_text().splitlines()]
    rec = next(r for r in records if sentence in r["messages"][0]["content"])
    # Byte-identical to the financial pack's eval prompt (which == the
    # module default), proving the generalization did not shift the output.
    expected = render_eval_prompt(
        sentence, template=load_task("financial_sentiment").eval_prompt
    )
    assert rec["messages"][0] == {"role": "user", "content": expected}
    assert rec["messages"][1] == {"role": "assistant", "content": "negative"}


def test_build_uses_task_label_names(tmp_path, monkeypatch):
    from types import SimpleNamespace

    fake_task = SimpleNamespace(
        labels=("World", "Sports"),
        label_names=("World", "Sports"),
        eval_prompt="topic: {text}",
    )
    monkeypatch.setattr(ds, "load_task", lambda _name: fake_task)
    rows = [(f"doc {i}", i % 2) for i in range(8)] + [("leaked doc", 0)]
    monkeypatch.setattr(ds, "load_source_examples", lambda _cfg: list(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: {"leaked doc"})

    result = ds.build_sft_dataset(cfg(tmp_path, val_fraction=0.0))

    recs = [json.loads(l) for l in result.train_path.read_text().splitlines()]
    assert recs
    assert all(r["messages"][1]["content"] in ("World", "Sports") for r in recs)
    assert all(r["messages"][0]["content"].startswith("topic: ") for r in recs)
    by_text = {
        r["messages"][0]["content"]: r["messages"][1]["content"] for r in recs
    }
    assert by_text["topic: doc 0"] == "World"
    assert by_text["topic: doc 1"] == "Sports"


def test_pool_below_min_raises(tmp_path, monkeypatch):
    rows = make_rows(2, 2, 2) + [LEAK_ROW]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)
    with pytest.raises(ds.LeakageError, match="pool"):
        ds.build_sft_dataset(cfg(tmp_path, min_pool_size=999))


def test_empty_eval_set_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "load_source_examples", fake_source(make_rows(3, 3, 3)))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())
    with pytest.raises(ds.LeakageError, match="empty"):
        ds.build_sft_dataset(cfg(tmp_path))


def test_zero_drop_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "load_source_examples", fake_source(make_rows(3, 3, 3)))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: {"a sentence not in the training pool"})
    with pytest.raises(ds.LeakageError, match="0 eval-set rows overlapped"):
        ds.build_sft_dataset(cfg(tmp_path))


def test_nfkc_divergent_text_still_matches(tmp_path, monkeypatch):
    # Pool sentence uses the "fi" ligature (U+FB01) and a full-width digit;
    # the eval set stores the plain ASCII form. NFKC in normalize_sentence
    # must collapse both so the leakage intersection still fires.
    rows = make_rows(3, 3, 3) + [("Proﬁt for １ unit rose", "positive")]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: {"profit for 1 unit rose"})
    result = ds.build_sft_dataset(cfg(tmp_path))
    assert result.dropped_count == 1


def test_min_pool_checked_after_neutral_balance(tmp_path, monkeypatch):
    # Pre-balance pool is 3+3+30 = 36 (>= min); balancing neutral down to the
    # 3-row minority class leaves 9 (< min) -- the guard must catch that.
    rows = make_rows(3, 3, 30) + [LEAK_ROW]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)
    with pytest.raises(ds.LeakageError, match="after neutral balancing"):
        ds.build_sft_dataset(cfg(tmp_path, balance_neutral=True, min_pool_size=20))


def test_deterministic_seeded_split(tmp_path, monkeypatch):
    rows = make_rows(20, 20, 20) + [LEAK_ROW]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)

    a = ds.build_sft_dataset(cfg(tmp_path / "a", seed=7, val_fraction=0.2))
    b = ds.build_sft_dataset(cfg(tmp_path / "b", seed=7, val_fraction=0.2))
    assert a.val_path.read_text() == b.val_path.read_text()
    train_texts = {json.loads(l)["messages"][0]["content"] for l in a.train_path.read_text().splitlines()}
    val_texts = {json.loads(l)["messages"][0]["content"] for l in a.val_path.read_text().splitlines()}
    assert train_texts.isdisjoint(val_texts)


def test_balance_neutral_downsamples(tmp_path, monkeypatch):
    rows = make_rows(5, 5, 40) + [LEAK_ROW]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)

    result = ds.build_sft_dataset(cfg(tmp_path, balance_neutral=True, val_fraction=0.0))
    assert result.train_class_counts["neutral"] == 5


def test_license_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "load_source_examples", fake_source(make_rows(2, 2, 2) + [LEAK_ROW]))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: LEAK_EVAL)
    result = ds.build_sft_dataset(cfg(tmp_path))
    assert "CC BY-NC-SA" in result.license_path.read_text()
    assert "Malo" in result.license_path.read_text()
