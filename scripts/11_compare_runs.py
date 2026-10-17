"""Compare caption metrics of several runs on the same COCO images, with paired
bootstrap intervals. Runs are directory names inside runs_dir (or full paths);
the result is also saved to <runs_dir>/caption_comparison.json.

    python scripts/11_compare_runs.py main_clip_qwen1.5b <siglip run> <dinov2 run> --names clip siglip dinov2
"""

import os
import sys

from _common import RUNS, run

args = sys.argv[1:]
cut = next((i for i, a in enumerate(args) if a.startswith("--")), len(args))
runs = [r if os.path.isabs(r) else os.path.join(RUNS, r) for r in args[:cut]]
if len(runs) < 2:
    sys.exit(__doc__)
run(sys.executable, "-m", "minivlm.compare", *runs,
    "--out", os.path.join(RUNS, "caption_comparison.json"), *args[cut:])
