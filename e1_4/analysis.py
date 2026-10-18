"""Summary of E1.4: probe accuracy (what the encoder encodes) against MiniVLM's
accuracy (what the full model reports), per task and encoder."""

import json
import os
from itertools import combinations

from minivlm.metrics import bootstrap, option_correct


def run_metrics(result: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """Metrics for one encoder's results_<name>.json."""
    out = {}
    for task, r in result["tasks"].items():
        items = r["vlm"]
        answers = [it["answer"] for it in items]
        correct = {
            "probe_grid": [int(p == a) for p, a in zip(r["probe"]["grid"]["test_pred"], answers)],
            "probe_mean": [int(p == a) for p, a in zip(r["probe"]["mean"]["test_pred"], answers)],
            "vlm_sum": [option_correct(it, "image", "sum") for it in items],
            "vlm_mean": [option_correct(it, "image", "mean") for it in items],
            "blank_sum": [option_correct(it, "blank", "sum") for it in items],
            "blank_mean": [option_correct(it, "blank", "mean") for it in items],
        }
        m = {k: bootstrap(v, n_bootstrap, seed) for k, v in correct.items()}
        best_vlm = max(["vlm_sum", "vlm_mean"], key=lambda k: m[k]["mean"])
        m["best_vlm"] = best_vlm
        # Gap: encoder (grid probe) minus the model, giving the model its better score.
        m["gap"] = bootstrap([p - v for p, v in zip(correct["probe_grid"], correct[best_vlm])],
                             n_bootstrap, seed)
        n_opt = len(items[0]["options"])
        preds = [min(range(n_opt), key=lambda i, it=it: it["nll_image"][i]) for it in items]
        m["vlm_prediction_share"] = {items[0]["options"][i]: preds.count(i) / len(preds)
                                     for i in range(n_opt)}
        m["chance"] = 1 / n_opt
        m["n_test"] = len(items)
        m["probe_curve"] = {k: v for k, v in r["probe"]["grid"]["curve"].items()}
        m["probe_weight_decay"] = r["probe"]["grid"]["weight_decay"]
        m["_correct"] = correct
        out[task] = m
    return out


def summarize(results: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    per_run = {name: run_metrics(r, n_bootstrap, seed) for name, r in results.items()}
    tasks = list(next(iter(per_run.values())))
    pairs = {}
    for task in tasks:
        for a, b in combinations(per_run, 2):
            for k in ["probe_grid", "vlm_sum", "vlm_mean"]:
                ca, cb = per_run[a][task]["_correct"][k], per_run[b][task]["_correct"][k]
                pairs.setdefault(task, {})[f"{k}: {a} - {b}"] = bootstrap(
                    [x - y for x, y in zip(ca, cb)], n_bootstrap, seed)
    for m in per_run.values():
        for t in m.values():
            t.pop("_correct")
    return dict(per_run=per_run, pairs=pairs)


def _pct(ci: dict) -> str:
    return f"{100 * ci['mean']:5.1f} [{100 * ci['lo']:5.1f}, {100 * ci['hi']:5.1f}]"


def format_summary(summary: dict, tasks_meta: dict) -> str:
    lines = []
    for task, meta in tasks_meta.items():
        lines.append(f"\n=== {task}: {meta['question']}  options {meta['options']}")
        lines.append(f"    test: {meta['test_per_class']} per class ({len(meta['test'])} images); "
                     f"probe training: {len(meta['train'])} images")
        lines.append(f"    {'encoder':10s} {'probe (grid)':>20s} {'probe (mean)':>20s} "
                     f"{'MiniVLM (sum)':>20s} {'MiniVLM (mean)':>20s} {'no image':>20s}  gap: probe − MiniVLM")
        for name, per_task in summary["per_run"].items():
            m = per_task[task]
            blank = max(m["blank_sum"], m["blank_mean"], key=lambda c: c["mean"])
            clear = "clear" if m["gap"]["lo"] > 0 or m["gap"]["hi"] < 0 else "not clear"
            lines.append(f"    {name:10s} {_pct(m['probe_grid']):>20s} {_pct(m['probe_mean']):>20s} "
                         f"{_pct(m['vlm_sum']):>20s} {_pct(m['vlm_mean']):>20s} {_pct(blank):>20s}  "
                         f"{_pct(m['gap'])} {clear}")
        lines.append(f"    chance {100 * next(iter(summary['per_run'].values()))[task]['chance']:.1f}%")
        for name, per_task in summary["per_run"].items():
            m = per_task[task]
            share = ", ".join(f"{k}: {100 * v:.0f}%" for k, v in m["vlm_prediction_share"].items())
            curve = ", ".join(f"{k}: {100 * v:.1f}" for k, v in m["probe_curve"].items())
            lines.append(f"    {name:10s} MiniVLM answers ({share}); probe accuracy by training "
                         f"images per class ({curve})")
        for pair, ci in summary["pairs"].get(task, {}).items():
            if "probe_grid" in pair or "vlm" in pair:
                clear = "clear" if ci["lo"] > 0 or ci["hi"] < 0 else ""
                lines.append(f"      {pair:36s} {_pct(ci)} {clear}")
    lines.append("\nAccuracy in % with 95% bootstrap intervals over the (class-balanced) test images. "
                 "gap = grid probe minus MiniVLM's better score, paired per image.")
    return "\n".join(lines)


def plot(summary: dict, tasks_meta: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["per_run"])
    tasks = list(tasks_meta)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True)
    bars = [("probe_grid", "encoder (linear probe)"), (None, "MiniVLM (better score)"),
            ("blank", "MiniVLM, no image")]
    width = 0.26
    for ax, task in zip(axes, tasks):
        for j, (key, label) in enumerate(bars):
            vals, errs = [], [[], []]
            for name in names:
                m = summary["per_run"][name][task]
                if key is None:
                    ci = m[m["best_vlm"]]
                elif key == "blank":
                    ci = max(m["blank_sum"], m["blank_mean"], key=lambda c: c["mean"])
                else:
                    ci = m[key]
                vals.append(100 * ci["mean"])
                errs[0].append(100 * (ci["mean"] - ci["lo"]))
                errs[1].append(100 * (ci["hi"] - ci["mean"]))
            xs = [i + (j - 1) * width for i in range(len(names))]
            ax.bar(xs, vals, width, yerr=errs, capsize=2, label=label)
        chance = 100 * summary["per_run"][names[0]][task]["chance"]
        ax.axhline(chance, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("What the encoder encodes vs what MiniVLM reports (dashed: chance)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write(summary: dict, tasks_meta: dict, work_dir: str):
    with open(os.path.join(work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot(summary, tasks_meta, os.path.join(work_dir, "bottleneck_probe.png"))
