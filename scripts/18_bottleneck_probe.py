"""E1.4 bottleneck probe on the three trained MiniVLM runs (`probe` section of
config.yaml): linear probes on each frozen encoder's features against
MiniVLM's own answers to the same COCO questions (presence, count, position of
people). Resumable. Results: <runs_dir>/<probe.run_name>/summary.json and
bottleneck_probe.png. Extra arguments are passed to e1_4.run.
"""

import os
import sys

from _common import CFG, DATA, RUNS, run

p = CFG["probe"]
runs = [r if os.path.isabs(r) else os.path.join(RUNS, r) for r in p["runs"]]
run(sys.executable, "-m", "e1_4.run", "--work_dir", os.path.join(RUNS, p["run_name"]),
    "--data_dir", DATA, "--runs", *runs, "--names", *p["names"], *sys.argv[1:])
