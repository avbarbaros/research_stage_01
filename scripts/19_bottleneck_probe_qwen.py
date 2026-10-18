"""E1.4 probes on Qwen2.5-VL-7B (`probe.qwen` in config.yaml): the same tasks,
test images, probes and readouts as the MiniVLM runs, to see whether a fully
trained VLM with 2-D (M-RoPE) positions carries layout to its answer. Run
scripts/18_bottleneck_probe.py first (it builds tasks.json). Resumable.
Results in <runs_dir>/<probe.run_name>/: results_<name>.json, summary_<name>.json,
bottleneck_probe_<name>.png, information_by_stage_<name>.png (MiniVLM runs
included for comparison). Extra arguments are passed to e1_4.qwen.
"""

import os
import sys

from _common import CFG, DATA, RUNS, run

p = CFG["probe"]
run(sys.executable, "-m", "e1_4.qwen", "--work_dir", os.path.join(RUNS, p["run_name"]),
    "--data_dir", DATA, *sys.argv[1:])
