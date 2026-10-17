"""Compare several runs evaluated on the same COCO images.

    python -m minivlm.compare <run dir> <run dir> ... [--names clip siglip dinov2]

Uses whichever results every run has:
    captions.json             CIDEr of generated captions (eval caption)
    reference_likelihood.json loss of the reference captions (eval reference)
    options_<benchmark>.json  multiple-choice accuracy (eval options)
Scores are paired image by image, so the intervals on the differences account
for some images being harder than others for every model.
"""

import argparse
import json
import os

from .config import yaml_from_argv
from .benchmarks import BENCHMARKS
from .metrics import (format_ci, paired_comparison, paired_likelihood_comparison,
                      paired_options_comparison)


def _load(runs: dict, filename: str):
    paths = {name: os.path.join(run, filename) for name, run in runs.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        missing = [name for name, p in paths.items() if not os.path.exists(p)]
        print(f"\nskipping {filename}: missing for {', '.join(missing)}")
        return None
    data = {}
    for name, p in paths.items():
        with open(p) as f:
            data[name] = json.load(f)
    return data


def _print_block(metric: str, m: dict):
    print(f"\n{metric}")
    for name, ci in m["per_run"].items():
        print(f"  {name:24s} {format_ci(ci)}")
    for pair, ci in m["pairs"].items():
        clear = "clear" if ci["lo"] > 0 or ci["hi"] < 0 else "not clear"
        print(f"  {pair:24s} {format_ci(ci):30s} {clear}")


def main():
    parent, y = yaml_from_argv()
    ap = argparse.ArgumentParser(parents=[parent])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--names", nargs="+", help="labels for the runs (default: directory names)")
    ap.add_argument("--out", help="write the comparison as JSON here")
    ap.add_argument("--bootstrap_samples", type=int, default=y["eval"]["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=y["eval"]["seed"])
    args = ap.parse_args()

    names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(names) != len(args.runs) or len(set(names)) != len(names):
        ap.error("--names needs one distinct label per run")
    runs = dict(zip(names, args.runs))
    out = dict(runs=runs)

    captions = _load(runs, "captions.json")
    if captions:
        result = paired_comparison({n: d["captions"] for n, d in captions.items()},
                                   args.bootstrap_samples, args.seed)
        out["captions"] = result
        print(f"=== Generated captions: CIDEr, higher is better ===\n"
              f"{result['n_images']} images captioned by every run; "
              f"{result['n_finished_in_all']} finished in all runs")
        for metric, m in result["metrics"].items():
            _print_block(metric, m)

    likelihood = _load(runs, "reference_likelihood.json")
    if likelihood:
        result = paired_likelihood_comparison({n: d["per_image"] for n, d in likelihood.items()},
                                              args.bootstrap_samples, args.seed)
        out["reference_likelihood"] = result
        print(f"\n=== Reference captions: loss per token ===\n"
              f"{result['n_images']} images, {result['n_tokens']} reference tokens")
        for m in result["metrics"].values():
            _print_block(m["description"], m)
        print("\nloss: lower is better, so a negative difference means the first run is better;"
              "\ngap: higher is better (the correct image helps more)")

    for bench in BENCHMARKS:
        data = _load(runs, f"options_{bench}.json")
        if data:
            result = paired_options_comparison({n: d["items"] for n, d in data.items()},
                                               args.bootstrap_samples, args.seed)
            out[f"options_{bench}"] = result
            chance = next(iter(data.values()))["summary"]["chance"]
            print(f"\n=== {bench}: option likelihood, higher is better ===\n"
                  f"{result['n_items']} items; chance {chance:.3f} per question")
            for m in result["metrics"].values():
                _print_block(m["description"], m)

    print("\n(mean [95% bootstrap interval]; a difference is clear when its interval excludes 0)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
