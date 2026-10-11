"""Shared paths and helpers for the numbered E1.1 scripts.

Settings come from config.yaml (or `--config other.yaml`). The environment
variables DATA_DIR, RUNS_DIR and RUN_NAME override the paths section.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minivlm.config import yaml_from_argv  # noqa: E402

_, CFG = yaml_from_argv()
_paths = CFG["paths"]
DRIVE = "/content/drive"


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


DRIVE_MISSING = False
DATA = _abs(os.environ.get("DATA_DIR") or _paths["data_dir"])
RUNS = _abs(os.environ.get("RUNS_DIR") or _paths["runs_dir"])
if RUNS.startswith(DRIVE + "/") and not os.path.isdir(os.path.join(DRIVE, "MyDrive")):
    # Writing here would create a local directory that later blocks the Drive mount.
    DRIVE_MISSING = True
    RUNS = _abs(_paths["runs_dir_fallback"])
RUN = os.path.join(RUNS, os.environ.get("RUN_NAME") or _paths["run_name"])
SMOKE_RUN = os.path.join(RUNS, CFG["smoke"]["run_name"])


def run(*cmd, check=True):
    """Run a command from the repo root, echoing it first."""
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode:
        sys.exit(result.returncode)
    return result.returncode


def module(name, *args):
    """python -m <name> <args> <any extra arguments given to the calling script>

    Extra arguments come last, so they override the values set by the script
    and by config.yaml.
    """
    return run(sys.executable, "-m", name, *args, *sys.argv[1:])
