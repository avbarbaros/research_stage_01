"""Download and install VLMEvalKit for E1.3, then check that everything the
sweep needs imports. Safe to re-run (e.g. in every new Colab runtime).

    python scripts/15_setup_vlmevalkit.py
    python scripts/15_setup_vlmevalkit.py --flash_attn   # also try FlashAttention-2 (slow build)

Why not just `pip install -e VLMEvalKit`: its requirements miss rouge-score,
which its code imports, and a single package that fails to build would abort
the whole install. So VLMEvalKit is installed without dependencies, its
requirements are installed separately (leaving the runtime's torch alone, and
falling back to one-by-one installs if the bulk install fails), and any module
still missing at import time is installed on the spot. Where the system lacks
libGL, opencv-python is swapped for opencv-python-headless. Location and pinned
commit: `resolution` section of config.yaml.
"""

import argparse
import importlib
import os
import subprocess
import sys

from _common import CFG

# Provided by the runtime; reinstalling could replace its CUDA build.
SKIP = {"torch", "torchvision"}
# Import name -> pip name, where they differ.
PIP_NAME = {"rouge_score": "rouge-score", "cv2": "opencv-python", "qwen_vl_utils": "qwen-vl-utils",
            "Levenshtein": "levenshtein", "sklearn": "scikit-learn", "PIL": "pillow",
            "dotenv": "python-dotenv", "yaml": "pyyaml"}
EXTRA = ["qwen-vl-utils", "rouge-score"]


def sh(*cmd, check=True):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def pip(*args, check=True):
    return sh(sys.executable, "-m", "pip", "install", "-q", *args, check=check)


def requirement_name(line: str) -> str:
    for sep in "<>=!~;[ ":
        line = line.split(sep)[0]
    return line.strip()


def install_requirements(path: str):
    with open(path) as f:
        reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    reqs = [r for r in reqs if requirement_name(r).lower() not in SKIP] + EXTRA
    if pip(*reqs, check=False) == 0:
        return
    print("\nBulk install failed; installing one by one to find the culprit(s).", flush=True)
    failed = [r for r in reqs if pip(r, check=False) != 0]
    if failed:
        print(f"\nCould not install: {', '.join(failed)}. These are only needed by other "
              "benchmarks; the import check below shows whether E1.3 is affected.")


def import_check(max_rounds: int = 8):
    """Import what e1_3.sweep uses, installing any module that turns out to be missing."""
    code = ("from vlmeval.dataset import build_dataset; from vlmeval.vlm import Qwen2VLChat; "
            "from vlmeval.smp.file import get_intermediate_file_path, get_pred_file_format; "
            "import qwen_vl_utils, transformers; transformers.Qwen2_5_VLForConditionalGeneration")
    for _ in range(max_rounds):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode == 0:
            return True
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
        if "libGL" in err:  # full OpenCV needs a desktop library: use the headless build
            print("libGL missing: switching to opencv-python-headless", flush=True)
            sh(sys.executable, "-m", "pip", "uninstall", "-q", "-y", "opencv-python", check=False)
            if pip("--force-reinstall", "--no-deps", "opencv-python-headless", check=False) != 0:
                return False
            continue
        if "No module named" not in err:
            print(r.stderr[-3000:])
            return False
        module = err.split("No module named")[1].strip().strip("'\"").split(".")[0]
        print(f"missing module {module!r}", flush=True)
        if pip(PIP_NAME.get(module, module), check=False) != 0:
            return False
    return False


def main():
    r = CFG["resolution"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=r["vlmevalkit_dir"])
    ap.add_argument("--commit", default=r["vlmevalkit_commit"])
    ap.add_argument("--flash_attn", action="store_true",
                    help="also try to install flash-attn (compiles if no wheel matches; can take long)")
    ap.add_argument("--config", help="settings file (default: config.yaml)")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.dir, ".git")):
        sh("git", "clone", "https://github.com/open-compass/VLMEvalKit", args.dir)
    sh("git", "-C", args.dir, "fetch", "-q", "origin", args.commit, check=False)
    sh("git", "-C", args.dir, "checkout", "-q", args.commit)

    pip("--no-deps", "-e", args.dir)
    install_requirements(os.path.join(args.dir, "requirements.txt"))
    if args.flash_attn:
        if pip("flash-attn", "--no-build-isolation", check=False) != 0:
            print("flash-attn failed to install; the sweep will use PyTorch SDPA attention instead.")

    importlib.invalidate_caches()
    if not import_check():
        sys.exit("\nVLMEvalKit is installed but the import check failed (see above).")
    try:
        import flash_attn  # noqa: F401
        attn = "FlashAttention-2"
    except ImportError:
        attn = "PyTorch SDPA (flash-attn not installed; slower, same results)"
    head = subprocess.run(["git", "-C", args.dir, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"\nVLMEvalKit {head} ready in {args.dir}. Attention: {attn}.")
    print("Next: python scripts/16_resolution_quick_check.py")


if __name__ == "__main__":
    main()
