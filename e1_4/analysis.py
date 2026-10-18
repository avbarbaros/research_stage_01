"""Summary of E1.4: probe accuracy (what the encoder encodes) against MiniVLM's
accuracy (what the full model reports), per task and encoder.

MiniVLM is read out three ways:
    raw         the option with the lowest loss (sum or mean over its tokens)
    calibrated  loss scaled by a temperature plus a per-option offset, both
                fitted on a few training images per class. It removes a fixed
                preference for one answer ("Yes", "Left") without adding any
                image information, and is compared with the probe trained on
                the same number of images per class.
    AUC         binary tasks only: how well the loss difference between the two
                options ranks the test images, independent of any threshold.
"""

import json
import os
from itertools import combinations

import numpy as np
import torch

from minivlm.metrics import bootstrap, option_correct


def fit_option_bias(items: list, steps: int = 200, l2: float = 1e-3) -> tuple:
    """(temperature, offsets) so that argmin(temperature * nll - offsets) fits the
    answers of `items` (class-weighted cross-entropy). Offset 0 is fixed at 0.

    The fit runs on losses centred per option and scaled to unit spread, so it
    behaves the same whether the options differ by hundredths or by tens of
    nats; the result is converted back to the original units."""
    nll = torch.tensor([it["nll_image"] for it in items], dtype=torch.float64)
    y = torch.tensor([it["answer"] for it in items])
    k = nll.shape[1]
    centre = nll.mean(dim=0)
    scale = float((nll - centre).std().clamp(min=1e-8))
    z = (nll - centre) / scale
    weights = len(y) / (k * torch.bincount(y, minlength=k).double().clamp(min=1))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    off = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, off], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = -log_t.exp() * z + off
        loss = (torch.nn.functional.cross_entropy(logits, y, weight=weights)
                + l2 * ((off ** 2).sum() + log_t.exp().pow(2).sum()))
        loss.backward()
        return loss

    opt.step(closure)
    t_std = float(log_t.detach().exp())
    if t_std < 1e-3:  # the losses carry no usable signal (or point the wrong way): offsets only
        t_std = 0.0
    t = t_std / scale
    offsets = off.detach() + t * centre
    return t, (offsets - offsets[0]).tolist()


def calibrated_predictions(items: list, temperature: float, offsets: list) -> list:
    return [int(np.argmin([temperature * x - o for x, o in zip(it["nll_image"], offsets)]))
            for it in items]


def auc(scores, labels) -> float:
    """Probability that a random class-1 item scores higher than a random class-0 item."""
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def bootstrap_auc(scores, labels, n_samples: int = 5000, seed: int = 0, level: float = 0.95) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels)
    rng = np.random.default_rng(seed)
    stats = [auc(s[i], y[i]) for i in rng.integers(0, len(s), size=(n_samples, len(s)))]
    lo, hi = np.nanquantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return dict(mean=auc(s, y), lo=float(lo), hi=float(hi), n=len(s))


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

        if r.get("vlm_calib"):
            t, off = fit_option_bias(r["vlm_calib"])
            pred = calibrated_predictions(items, t, off)
            correct["vlm_calibrated"] = [int(p == a) for p, a in zip(pred, answers)]
            m["vlm_calibrated"] = bootstrap(correct["vlm_calibrated"], n_bootstrap, seed)
            m["calibration"] = dict(temperature=t, offsets=off, n_images=len(r["vlm_calib"]))
            m["vlm_calibrated_share"] = {items[0]["options"][i]: pred.count(i) / len(pred)
                                         for i in range(n_opt)}
            # Same number of training images per class on both sides.
            n_cal = len(r["vlm_calib"]) // n_opt
            probe_small = r["probe"]["grid"].get("curve_pred", {}).get(str(n_cal))
            if probe_small:
                c = [int(p == a) for p, a in zip(probe_small, answers)]
                m["probe_grid_small"] = bootstrap(c, n_bootstrap, seed)
                m["probe_grid_small"]["per_class"] = n_cal
                m["gap_calibrated"] = bootstrap(
                    [p - v for p, v in zip(c, correct["vlm_calibrated"])], n_bootstrap, seed)
        if n_opt == 2:
            score = r["probe"]["grid"].get("test_score")
            if score:
                m["auc_probe"] = bootstrap_auc(score, answers, n_bootstrap, seed)
            m["auc_vlm"] = bootstrap_auc([it["nll_image"][0] - it["nll_image"][1] for it in items],
                                         answers, n_bootstrap, seed)

        if r.get("layers"):
            lay = r["layers"]
            m["stages"] = {k: bootstrap([int(p == a) for p, a in zip(v["test_pred"], lay["_answers"])],
                                        n_bootstrap, seed)
                           for k, v in lay.items() if k != "_answers"}
        m["_correct"] = correct
        out[task] = m
    return out


def summarize(results: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    per_run = {name: run_metrics(r, n_bootstrap, seed) for name, r in results.items()}
    tasks = list(next(iter(per_run.values())))
    pairs = {}
    for task in tasks:
        for a, b in combinations(per_run, 2):
            for k in ["probe_grid", "vlm_sum", "vlm_mean", "vlm_calibrated"]:
                if k not in per_run[a][task]["_correct"]:
                    continue
                ca, cb = per_run[a][task]["_correct"][k], per_run[b][task]["_correct"][k]
                pairs.setdefault(task, {})[f"{k}: {a} - {b}"] = bootstrap(
                    [x - y for x, y in zip(ca, cb)], n_bootstrap, seed)
    for m in per_run.values():
        for t in m.values():
            t.pop("_correct")
    return dict(per_run=per_run, pairs=pairs)


def _clear(ci: dict) -> str:
    return "clear" if ci["lo"] > 0 or ci["hi"] < 0 else "not clear"


def _stage_label(name: str) -> str:
    if name == "encoder":
        return "enc"
    kind, layer = name.split("_L")
    return f"{'img' if kind == 'img' else 'ans'} L{layer}"


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
        for name, per_task in summary["per_run"].items():
            m = per_task[task]
            parts = []
            if "vlm_calibrated" in m:
                share = ", ".join(f"{k}: {100 * v:.0f}%" for k, v in m["vlm_calibrated_share"].items())
                parts.append(f"calibrated MiniVLM {_pct(m['vlm_calibrated'])} ({share})")
            if "gap_calibrated" in m:
                g = m["gap_calibrated"]
                parts.append(f"probe at {m['probe_grid_small']['per_class']}/class "
                             f"{_pct(m['probe_grid_small'])}, gap {_pct(g)} {_clear(g)}")
            if "auc_vlm" in m:
                parts.append(f"AUC probe {_pct(m['auc_probe'])}, MiniVLM {_pct(m['auc_vlm'])}"
                             if "auc_probe" in m else f"AUC MiniVLM {_pct(m['auc_vlm'])}")
            if parts:
                lines.append(f"    {name:10s} " + "; ".join(parts))
        for name, per_task in summary["per_run"].items():
            stages = per_task[task].get("stages")
            if stages:
                if name == next(iter(summary["per_run"])):
                    lines.append("    probe accuracy by stage (same images for every stage):")
                lines.append(f"    {name:10s} " + "  ".join(
                    f"{_stage_label(k)} {100 * v['mean']:.1f}" for k, v in stages.items()))
        for pair, ci in summary["pairs"].get(task, {}).items():
            if "probe_grid" in pair or "vlm" in pair:
                clear = "clear" if ci["lo"] > 0 or ci["hi"] < 0 else ""
                lines.append(f"      {pair:36s} {_pct(ci)} {clear}")
    lines.append("\nAccuracy in % with 95% bootstrap intervals over the (class-balanced) test images. "
                 "gap = grid probe minus MiniVLM's better score, paired per image. Calibrated "
                 "MiniVLM: temperature and per-option offset fitted on a few training images per "
                 "class, compared with the probe trained on as many. Stages: enc = encoder grid "
                 "probe, img Lk = image-token states at LLM layer k (L0 = projector output), "
                 "ans Lk = answer-position state at layer k.")
    return "\n".join(lines)


def plot(summary: dict, tasks_meta: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["per_run"])
    tasks = list(tasks_meta)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True)
    bars = [("probe_grid", "encoder (linear probe)"), (None, "MiniVLM (better score)"),
            ("vlm_calibrated", "MiniVLM, calibrated"), ("blank", "MiniVLM, no image")]
    if "vlm_calibrated" not in summary["per_run"][names[0]][tasks[0]]:
        bars.pop(2)
    width = 0.8 / len(bars)
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
            xs = [i + (j - (len(bars) - 1) / 2) * width for i in range(len(names))]
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


def plot_stages(summary: dict, tasks_meta: dict, path: str) -> bool:
    """Probe accuracy along the model, one panel per task, one line per encoder."""
    names = list(summary["per_run"])
    tasks = [t for t in tasks_meta if summary["per_run"][names[0]][t].get("stages")]
    if not tasks:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8), sharey=True,
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for ci, name in enumerate(names):
            m = summary["per_run"][name][task]
            st = m["stages"]
            img = ["encoder"] + [k for k in st if k.startswith("img_")]
            ans = [k for k in st if k.startswith("last_")]
            color = f"C{ci}"
            ax.plot(range(len(img)), [100 * st[k]["mean"] for k in img], "o-", color=color,
                    label=f"{name}: image tokens")
            ax.plot(range(1, len(ans) + 1), [100 * st[k]["mean"] for k in ans], "s--", color=color,
                    alpha=0.7, label=f"{name}: answer position")
            final = m.get("vlm_calibrated", m[m["best_vlm"]])
            ax.plot([len(img)], [100 * final["mean"]], "*", color=color, markersize=12)
        labels = ["enc"] + [_stage_label(k).replace("img ", "") for k in img[1:]] + ["answer"]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(100 * summary["per_run"][names[0]][task]["chance"], color="grey",
                   linestyle=":", linewidth=1)
        ax.set_title(task)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("linear probe accuracy (%)")
    axes[0][0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Where the information is lost (L0 = projector output; star: MiniVLM's answer, "
                 "calibrated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def write(summary: dict, tasks_meta: dict, work_dir: str):
    with open(os.path.join(work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot(summary, tasks_meta, os.path.join(work_dir, "bottleneck_probe.png"))
    plot_stages(summary, tasks_meta, os.path.join(work_dir, "information_by_stage.png"))
