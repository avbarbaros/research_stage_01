"""Print the main run's results and save an image + caption grid to
<run>/captions_grid.png."""

import json
import os
import textwrap

from _common import DATA, RUN

from minivlm.metrics import format_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

with open(os.path.join(RUN, "captions.json")) as f:
    result = json.load(f)
grounding_path = os.path.join(RUN, "grounding.json")
if os.path.exists(grounding_path):
    with open(grounding_path) as f:
        print("grounding loss:", json.load(f))
if "metrics" in result:
    print(format_metrics(result["metrics"]))
else:
    print("CIDEr (short prompt):", result.get("cider_short"),
          "\n(older captions.json: run `python -m minivlm.eval rescore --out_dir <run>` for all metrics)")

caps = result["captions"][:12]
fig, axes = plt.subplots(3, 4, figsize=(18, 15))
for ax in axes.flat:
    ax.axis("off")
for ax, c in zip(axes.flat, caps):
    ax.imshow(Image.open(os.path.join(DATA, "val2017", c["image"])))
    ax.set_title(textwrap.fill(c["short"], 42), fontsize=10)
plt.tight_layout()
out = os.path.join(RUN, "captions_grid.png")
plt.savefig(out, dpi=100)
print(f"saved {out}")

for c in caps:
    print(f"\n{c['image']}")
    print("  short : " + c["short"])
    if "detail" in c:
        print("  detail: " + textwrap.fill(c["detail"], 110, subsequent_indent=" " * 10))
    print("  ref   : " + c["references"][0])
