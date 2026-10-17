"""Projector-size ablation (E1.1 gate: "know what breaks when it is too small").

Trains the projector in several sizes on the first 10k samples of the main
subset (same data and order for every size), evaluates each, and compares
them:
    mlp2_1536  Linear 1024->1536, GELU, Linear 1536->1536   (the default)
    mlp2_64    the same with a 64-wide hidden layer         (bottleneck)
    mlp2_16    the same with a 16-wide hidden layer         (severe bottleneck)
    linear     one Linear 1024->1536, no non-linearity      (LLaVA-1)

Safe to re-run after a disconnect: training resumes and finished evaluations
are skipped. Needs the main subset (05_prepare_data.py) on local disk.

    python scripts/14_projector_ablation.py
    python scripts/14_projector_ablation.py --variants linear mlp2_64 --max_samples 5000
Other arguments (e.g. --batch_size 4 --accum 8) are passed to minivlm.train.
"""

import argparse
import csv
import json
import os
import sys

from _common import CFG, DATA, RUNS, run

VARIANTS = {
    "mlp2_1536": ["--projector", "mlp2", "--projector_hidden", "0"],
    "mlp2_64": ["--projector", "mlp2", "--projector_hidden", "64"],
    "mlp2_16": ["--projector", "mlp2", "--projector_hidden", "16"],
    "linear": ["--projector", "linear"],
}

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
ap.add_argument("--max_samples", type=int, default=10000)
ap.add_argument("--encoder", default=CFG["model"]["encoder"])
ap.add_argument("--eval_items", type=int, default=0,
                help="first N MMVP / A-OKVQA items only, for a quick check (0 = all)")
ap.add_argument("--config", help="settings file (default: config.yaml)")
args, train_extra = ap.parse_known_args()
config = ["--config", args.config] if args.config else []
py = [sys.executable, "-m"]


def projector_params(variant: str) -> int:
    from transformers import AutoConfig

    from minivlm.config import ENCODERS

    enc = AutoConfig.from_pretrained(ENCODERS[args.encoder]["model_id"])
    v = getattr(enc, "vision_config", enc).hidden_size
    d = AutoConfig.from_pretrained(CFG["model"]["llm"]).hidden_size
    if variant == "linear":
        return v * d + d
    h = int(VARIANTS[variant][-1]) or d
    return v * h + h + h * d + d


runs = {}
for variant in args.variants:
    out = os.path.join(RUNS, f"abl{args.max_samples // 1000}k_{args.encoder}_{variant}")
    runs[variant] = out
    done = lambda name: os.path.exists(os.path.join(out, name))  # noqa: E731
    print(f"\n######## {variant}: {out}", flush=True)
    if not done("grounding.json"):
        run(*py, "minivlm.train", "--data_dir", DATA, "--out_dir", out, "--encoder", args.encoder,
            "--max_samples", args.max_samples, "--resume", *VARIANTS[variant], *config,
            *train_extra)
        run(*py, "minivlm.eval", "grounding", "--data_dir", DATA, "--out_dir", out, *config)
    if not done("reference_likelihood.json"):
        run(*py, "minivlm.eval", "reference", "--data_dir", DATA, "--out_dir", out, *config)
    if not (done("options_mmvp.json") and done("options_aokvqa.json")):
        run(*py, "minivlm.eval", "options", "--out_dir", out, "--max_items", args.eval_items,
            *config)

# Paired comparison with intervals
run(*py, "minivlm.compare", *runs.values(), "--names", *runs,
    "--out", os.path.join(RUNS, f"projector_ablation_{args.encoder}.json"), *config)


# One-line-per-variant summary and a combined loss curve
def last_losses(out, k=5):
    with open(os.path.join(out, "train_log.csv")) as f:
        rows = list(csv.DictReader(f))
    return [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows], \
        sum(float(r["loss"]) for r in rows[-k:]) / min(k, len(rows))


print(f"\n{'variant':10s} {'params':>9s} {'train':>6s} {'held-out':>8s} {'gap':>6s} "
      f"{'blank gap':>9s} {'MMVP pair':>9s} {'A-OKVQA':>7s}")
curves = {}
for variant, out in runs.items():
    steps, losses, final = last_losses(out)
    curves[variant] = (steps, losses)
    with open(os.path.join(out, "grounding.json")) as f:
        g = json.load(f)
    with open(os.path.join(out, "options_mmvp.json")) as f:
        mmvp = json.load(f)["summary"]
    with open(os.path.join(out, "options_aokvqa.json")) as f:
        aok = json.load(f)["summary"]
    print(f"{variant:10s} {projector_params(variant):9,d} {final:6.3f} {g['correct']:8.3f} "
          f"{g['shuffled'] - g['correct']:6.3f} {g['blank'] - g['correct']:9.3f} "
          f"{mmvp['pair_acc_sum']['mean']:9.3f} {aok['acc_sum']['mean']:7.3f}")
print("train: mean of the last 5 logged losses; held-out: loss per token with the correct image;"
      "\ngap / blank gap: how much worse the loss gets with a wrong / blank image (higher = more grounded);"
      "\nMMVP pair accuracy and A-OKVQA accuracy with the summed-loss score. Intervals: see above.")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.figure(figsize=(7, 3.8))
for variant, (steps, losses) in curves.items():
    plt.plot(steps, losses, linewidth=1.2, label=f"{variant} ({projector_params(variant):,} params)")
plt.xlabel("optimiser step")
plt.ylabel("training loss")
plt.title(f"Projector size, {args.max_samples:,} samples, {args.encoder}")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
path = os.path.join(RUNS, f"projector_ablation_{args.encoder}_loss.png")
plt.savefig(path, dpi=150)
print(f"saved {path}")
