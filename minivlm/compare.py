"""Compare caption metrics of several runs on the same COCO images.

    python -m minivlm.compare <run dir> <run dir> ... [--names clip siglip dinov2]

Scores are paired image by image, so the intervals on the differences account
for some images being harder than others for every model. Works on old
captions.json files too (finished captions are then detected by punctuation).
"""

import argparse
import json
import os

from .config import yaml_from_argv
from .metrics import format_ci, paired_comparison


def main():
    parent, y = yaml_from_argv()
    ap = argparse.ArgumentParser(parents=[parent])
    ap.add_argument("runs", nargs="+", help="run directories containing captions.json")
    ap.add_argument("--names", nargs="+", help="labels for the runs (default: directory names)")
    ap.add_argument("--out", help="write the comparison as JSON here")
    ap.add_argument("--bootstrap_samples", type=int, default=y["eval"]["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=y["eval"]["seed"])
    args = ap.parse_args()

    names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(names) != len(args.runs) or len(set(names)) != len(names):
        ap.error("--names needs one distinct label per run")
    runs = {}
    for name, run in zip(names, args.runs):
        with open(os.path.join(run, "captions.json")) as f:
            runs[name] = json.load(f)["captions"]

    result = paired_comparison(runs, args.bootstrap_samples, args.seed)
    n = result["n_images"]
    print(f"{n} images captioned by every run; {result['n_finished_in_all']} finished in all runs")
    for metric, m in result["metrics"].items():
        print(f"\n{metric}")
        for name, ci in m["per_run"].items():
            print(f"  {name:24s} {format_ci(ci)}")
        for pair, ci in m["pairs"].items():
            clear = "clear" if ci["lo"] > 0 or ci["hi"] < 0 else "not clear"
            print(f"  {pair:24s} {format_ci(ci):30s} {clear}")
    print("\n(mean [95% bootstrap interval]; a difference is clear when its interval excludes 0)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(runs=dict(zip(names, args.runs)), **result), f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
