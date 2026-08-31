from app.config.tasks import load_task
from app.data.loader import load_task_examples


def test_ag_news_pack_loads_and_is_balanced():
    task = load_task("ag_news")
    examples = load_task_examples(task)
    assert len(examples) == 120
    counts = {}
    for e in examples:
        counts[e.gold_label] = counts.get(e.gold_label, 0) + 1
    assert set(counts) == {"World", "Sports", "Business", "Sci/Tech"}
    assert all(c == 30 for c in counts.values())
