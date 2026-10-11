"""Save the training-loss curve of the main run to <run>/loss.png."""

import csv
import os

from _common import RUN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(os.path.join(RUN, "train_log.csv")) as f:
    rows = list(csv.DictReader(f))
steps, loss = [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows]

plt.figure(figsize=(7, 3.5))
plt.plot(steps, loss, linewidth=1)
plt.xlabel("optimiser step")
plt.ylabel("training loss")
plt.title("Projector training")
plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(RUN, "loss.png")
plt.savefig(out, dpi=150)
print(f"first logged loss {loss[0]:.3f} (step {steps[0]}), last {loss[-1]:.3f} (step {steps[-1]})")
print(f"saved {out}")
