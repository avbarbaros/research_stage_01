"""Accuracy against visual tokens, per task category, from a finished sweep.

    python -m e1_3.analysis --work_dir runs/e1_3

Reads every tokens_*/<dataset>/scores.json, then for each task category
(each dataset overall, and each MMBench ability group) reports accuracy with a
95% bootstrap interval at every budget, and the point of diminishing returns:
the smallest budget whose accuracy is not clearly below the best budget's
(paired bootstrap over the same questions). Writes summary.json and
resolution_sweep.png to the work directory.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

from minivlm.metrics import bootstrap


def load_scores(work_dir: str) -> dict:
    """{(dataset, budget): items}"""
    out = {}
    for path in glob.glob(os.path.join(work_dir, "tokens_*", "*", "scores.json")):
        with open(path) as f:
            r = json.load(f)
        out[(r["dataset"], r["budget"])] = r["items"]
    return out


def summarize(scores: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    # category -> budget -> {item id: (score, tokens, native, seconds)}
    table = defaultdict(lambda: defaultdict(dict))
    for (dataset, budget), items in scores.items():
        for idx, it in items.items():
            row = (it["score"], it["tokens"], it["native_tokens"], it["seconds"])
            table[dataset][budget][idx] = row
            if it["group"] != dataset:
                table[it["group"]][budget][idx] = row

    summary = {}
    for category, by_budget in sorted(table.items()):
        budgets = sorted(by_budget)
        common = sorted(set.intersection(*(set(by_budget[b]) for b in budgets)))
        if not common:
            continue

        def col(b, k):
            return [by_budget[b][i][k] for i in common]

        points = {}
        for b in budgets:
            points[b] = dict(accuracy=bootstrap(col(b, 0), n_bootstrap, seed),
                             mean_tokens=sum(col(b, 1)) / len(common),
                             mean_seconds=sum(col(b, 3)) / len(common))
        best = max(budgets, key=lambda b: points[b]["accuracy"]["mean"])
        for b in budgets:
            points[b]["vs_best"] = bootstrap([x - y for x, y in zip(col(b, 0), col(best, 0))],
                                             n_bootstrap, seed)
        enough = next(b for b in budgets if points[b]["vs_best"]["hi"] >= 0)
        summary[category] = dict(
            n_items=len(common),
            mean_native_tokens=sum(col(budgets[0], 2)) / len(common),
            best_budget=best,
            diminishing_returns_budget=enough,
            points={str(b): p for b, p in points.items()},
        )
    return summary


def _interpolate(x: float, xs: list, ys: list) -> float:
    """Piecewise-linear in x, extended linearly beyond the measured range."""
    if len(xs) == 1:
        return ys[0]
    i = 0 if x <= xs[0] else min(len(xs) - 2, max(k for k in range(len(xs) - 1) if xs[k] <= x))
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return max(min(ys), ys[i] + slope * (x - xs[i]))


def estimate_runtime(work_dir: str, budgets: list, n_per_dataset: int) -> dict:
    """Hours a full sweep would take, extrapolated from the seconds per generated
    answer measured in this (smaller) sweep. Model loading is not included."""
    measured = defaultdict(dict)  # dataset -> mean tokens -> (seconds per row, rows per group)
    for path in glob.glob(os.path.join(work_dir, "tokens_*", "*", "scores.json")):
        with open(path) as f:
            r = json.load(f)
        with open(os.path.join(os.path.dirname(path), "predictions.json")) as f:
            preds = json.load(f)
        tokens = sum(it["tokens"] for it in r["items"].values()) / len(r["items"])
        seconds = sum(p["seconds"] for p in preds.values()) / len(preds)
        measured[r["dataset"]][tokens] = (seconds, len(preds) / len(r["items"]))
    hours = {}
    for dataset, points in measured.items():
        xs = sorted(points)
        secs = [points[x][0] for x in xs]
        rows = n_per_dataset * sum(points[x][1] for x in xs) / len(xs)
        hours[dataset] = rows * sum(_interpolate(b, xs, secs) for b in budgets) / 3600
    return dict(total_hours=sum(hours.values()), per_dataset_hours=hours)


def format_summary(summary: dict) -> str:
    lines = []
    for category, s in summary.items():
        lines.append(f"\n{category}  ({s['n_items']} questions; native resolution "
                     f"≈{s['mean_native_tokens']:.0f} tokens)")
        for b, p in s["points"].items():
            a, d = p["accuracy"], p["vs_best"]
            mark = " <- best" if int(b) == s["best_budget"] else (
                " <- enough" if int(b) == s["diminishing_returns_budget"] else "")
            lines.append(f"  {int(b):5d} tokens (mean {p['mean_tokens']:6.0f})  "
                         f"acc {100 * a['mean']:5.1f} [{100 * a['lo']:5.1f}, {100 * a['hi']:5.1f}]  "
                         f"vs best {100 * d['mean']:+5.1f} [{100 * d['lo']:+5.1f}, {100 * d['hi']:+5.1f}]  "
                         f"{p['mean_seconds']:.2f} s/q{mark}")
    lines.append("\n'enough' = smallest budget not clearly below the best (paired 95% bootstrap); "
                 "accuracy in %, MMBench with circular evaluation.")
    return "\n".join(lines)


def plot(summary: dict, path: str, datasets: list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[3, 2]))
    for category, s in summary.items():
        xs = [p["mean_tokens"] for p in s["points"].values()]
        ys = [100 * p["accuracy"]["mean"] for p in s["points"].values()]
        overall = category in datasets
        ax.plot(xs, ys, marker="o", linewidth=2.2 if overall else 1.0,
                alpha=1.0 if overall else 0.7, linestyle="-" if overall else "--",
                label=category)
        if overall:
            ax_t.plot(xs, [p["mean_seconds"] for p in s["points"].values()], marker="o",
                      linewidth=1.8, label=category)
    for a in (ax, ax_t):
        a.set_xscale("log", base=2)
        a.set_xlabel("visual tokens per image")
        a.grid(alpha=0.3)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Qwen2.5-VL-7B: accuracy vs visual tokens")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax_t.set_ylabel("seconds per question")
    ax_t.set_title("Cost")
    ax_t.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    from minivlm.config import yaml_from_argv

    parent, y = yaml_from_argv()
    ap = argparse.ArgumentParser(parents=[parent], description=__doc__.split("\n")[0])
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--bootstrap_samples", type=int, default=y["eval"]["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=y["eval"]["seed"])
    args = ap.parse_args(argv)

    scores = load_scores(args.work_dir)
    if not scores:
        raise SystemExit(f"no scores.json under {args.work_dir}")
    summary = summarize(scores, args.bootstrap_samples, args.seed)
    print(format_summary(summary))
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    path = os.path.join(args.work_dir, "resolution_sweep.png")
    plot(summary, path, sorted({d for d, _ in scores}))
    print(f"\nsaved {path} and summary.json")


if __name__ == "__main__":
    main()
