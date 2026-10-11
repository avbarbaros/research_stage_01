"""Check the GPU, install dependencies, and show where data and runs will go."""

import os
import shutil
import sys

from _common import DATA, DRIVE_MISSING, ROOT, RUN, RUNS, run

if shutil.which("nvidia-smi"):
    run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
else:
    print("no nvidia-smi found: training will run on CPU")

run(sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(ROOT, "requirements.txt"))

os.makedirs(DATA, exist_ok=True)
os.makedirs(RUNS, exist_ok=True)
print(f"\ndata dir : {DATA}")
print(f"runs dir : {RUNS}")
print(f"main run : {RUN}")
if DRIVE_MISSING:
    print("\nGoogle Drive is not mounted, so runs use paths.runs_dir_fallback and are lost when "
          "the runtime ends.\nDrive can only be mounted from a notebook cell:\n"
          "    from google.colab import drive; drive.mount('/content/drive')\n"
          "or set paths.runs_dir in config.yaml to another persistent location.")
