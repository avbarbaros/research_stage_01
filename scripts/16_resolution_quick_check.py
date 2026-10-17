"""Quick check before the full E1.3 sweep: a small sweep (two budgets, a few
questions per dataset; `quick_check_*` in config.yaml) through the exact same
code, then an estimate of how long the full sweep will take on this GPU.
Results go to <runs_dir>/<run_name>_check, separate from the real sweep.

    python scripts/16_resolution_quick_check.py
    python scripts/16_resolution_quick_check.py --n 10 --datasets MMBench_DEV_EN
Extra arguments are passed to e1_3.sweep.
"""

import os
import sys

from _common import CFG, RUNS, run

r = CFG["resolution"]
work_dir = os.path.join(RUNS, f"{r['run_name']}_check")
args = sys.argv[1:]
config = args[args.index("--config"): args.index("--config") + 2] if "--config" in args else []

run(sys.executable, "-m", "e1_3.sweep", "--work_dir", work_dir,
    "--budgets", *map(str, r["quick_check_budgets"]), "--n", str(r["quick_check_n"]), *args)
run(sys.executable, "-m", "e1_3.analysis", "--work_dir", work_dir, *config)

from e1_3.analysis import estimate_runtime  # noqa: E402

est = estimate_runtime(work_dir, r["budgets"], r["n_per_dataset"])
print(f"\nEstimated full sweep ({len(r['budgets'])} budgets {r['budgets']}, "
      f"{r['n_per_dataset']} questions per dataset): {est['total_hours']:.1f} h of generation")
for dataset, h in est["per_dataset_hours"].items():
    print(f"  {dataset:16s} {h:5.1f} h")
print("Extrapolated from this check's seconds per answer; model loading and dataset "
      "downloads come on top.\nNext: python scripts/17_resolution_sweep.py")
