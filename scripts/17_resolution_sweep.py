"""E1.3: Qwen2.5-VL-7B at several visual-token budgets on VLMEvalKit datasets
(`resolution` section of config.yaml), then accuracy against tokens per task
category. Resumable: re-run after a disconnect. Run 15_setup_vlmevalkit.py
first, and 16_resolution_quick_check.py to see how long this will take.
Extra arguments are passed to e1_3.sweep.
"""

import os
import sys

from _common import CFG, RUNS, run

work_dir = os.path.join(RUNS, os.environ.get("E1_3_RUN_NAME") or CFG["resolution"]["run_name"])
run(sys.executable, "-m", "e1_3.sweep", "--work_dir", work_dir, *sys.argv[1:])
args = sys.argv[1:]
config = args[args.index("--config"): args.index("--config") + 2] if "--config" in args else []
run(sys.executable, "-m", "e1_3.analysis", "--work_dir", work_dir, *config)
