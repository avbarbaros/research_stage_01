import pytest

from e1_3.analysis import summarize
from e1_3.subsets import sample_indices
from e1_3.tokens import budget_pixels, native_tokens, visual_tokens


@pytest.mark.parametrize("h,w", [(224, 224), (480, 640), (1080, 1920), (300, 2000), (50, 60)])
@pytest.mark.parametrize("budget", [64, 256, 1024])
def test_budget_fixes_token_count(h, w, budget):
    tokens = visual_tokens(h, w, budget_pixels(budget), budget_pixels(budget))
    # downscaled images round down, upscaled ones round up; extreme aspect ratios drift most
    assert 0.6 * budget <= tokens <= 1.6 * budget


def test_known_token_counts():
    assert native_tokens(224, 224) == 64          # 8 x 8 tokens of 28 px
    assert native_tokens(448, 672) == 16 * 24
    assert visual_tokens(1000, 1000, budget_pixels(256), budget_pixels(256)) == 256


def test_mmbench_subset_keeps_whole_groups():
    # three questions with 2, 3 and 4 rotated copies
    idx = [1, 1_000_001, 2, 1_000_002, 2_000_002, 3, 1_000_003, 2_000_003, 3_000_003]
    rows = sample_indices("MMBench_DEV_EN", idx, n=2, seed=0)
    groups = {i % 1_000_000 for i in rows}
    assert len(groups) == 2
    assert rows == [i for i in idx if i % 1_000_000 in groups]
    assert rows == sample_indices("MMBench_DEV_EN", idx, n=2, seed=0)
    assert len(sample_indices("TextVQA_VAL", list(range(100)), n=10)) == 10
    assert len(sample_indices("TextVQA_VAL", list(range(5)), n=0)) == 5


def test_summarize_categories_and_diminishing_returns():
    def items(scores, group="MMBench_DEV_EN: perception", tokens=64):
        return {str(i): dict(score=s, group=group, tokens=tokens, native_tokens=300, seconds=0.5)
                for i, s in enumerate(scores)}
    low = [0] * 20 + [1] * 20                      # 50%
    high = [1] * 40                                # 100%
    almost = [0] + [1] * 39                        # 97.5%
    scores = {("MMBench_DEV_EN", 64): items(low, tokens=64),
              ("MMBench_DEV_EN", 256): items(almost, tokens=256),
              ("MMBench_DEV_EN", 1024): items(high, tokens=1024)}
    s = summarize(scores, n_bootstrap=500)
    assert set(s) == {"MMBench_DEV_EN", "MMBench_DEV_EN: perception"}
    m = s["MMBench_DEV_EN"]
    assert m["best_budget"] == 1024
    assert m["diminishing_returns_budget"] == 256   # 1 question behind: not clearly worse
    assert m["points"]["64"]["vs_best"]["hi"] < 0
    assert m["points"]["256"]["mean_tokens"] == 256


def test_interpolate_and_runtime_estimate(tmp_path):
    import json
    from e1_3.analysis import _interpolate, estimate_runtime
    assert _interpolate(500, [100, 900], [1.0, 3.0]) == 2.0
    assert _interpolate(1700, [100, 900], [1.0, 3.0]) == 5.0     # extended beyond the range
    assert _interpolate(10, [100, 900], [1.0, 3.0]) == 1.0       # never below the fastest measured

    # MMBench-like: 2 question groups, 3 rows each, 2 s per row at ~64 tokens, 4 s at ~1024
    for budget, secs in [(64, 2.0), (1024, 4.0)]:
        d = tmp_path / f"tokens_{budget:05d}" / "MMBench_DEV_EN"
        d.mkdir(parents=True)
        items = {str(i): dict(score=1, group="g", tokens=budget, native_tokens=200, seconds=secs)
                 for i in range(2)}
        json.dump(dict(dataset="MMBench_DEV_EN", budget=budget, n_items=2, items=items),
                  open(d / "scores.json", "w"))
        json.dump({str(i): dict(prediction="A", seconds=secs) for i in range(6)},
                  open(d / "predictions.json", "w"))
    est = estimate_runtime(str(tmp_path), budgets=[64, 1024], n_per_dataset=100)
    # 100 groups x 3 rows x (2 + 4) s = 1800 s
    assert abs(est["total_hours"] - 1800 / 3600) < 1e-9
