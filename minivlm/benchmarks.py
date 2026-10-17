"""Multiple-choice benchmarks scored by option likelihood.

Each item: {id, question, options, answer (index), image, group}. `image` is a
file path or raw bytes; `group` ties the two questions of an MMVP pair.

MMVP (Tong et al., "Eyes Wide Shut?", 2024): 150 pairs of images that CLIP
embeds almost identically but that differ visibly. Both questions of a pair
are the same, with the same two options; only the image differs and the
correct answers are opposite. Pair accuracy (both right) is the standard
metric: a model that ignores the image gets at most one of the two right.

A-OKVQA (Schwenk et al., 2022), validation split: 1,145 questions on COCO
val2017 images, four choices each, often needing commonsense knowledge.
"""

import csv
import io
import os
import re

from PIL import Image

_OPTION = re.compile(r"\(([a-z])\)\s*(.*?)\s*(?=\([a-z]\)|$)")


def parse_mmvp_options(text: str) -> list:
    """'(a) Open (b) Closed' -> ['Open', 'Closed']"""
    return [m.group(2) for m in _OPTION.finditer(text.strip())]


def load_mmvp() -> list:
    from huggingface_hub import snapshot_download

    root = snapshot_download("MMVP/MMVP", repo_type="dataset",
                             allow_patterns=["Questions.csv", "MMVP Images/*"])
    with open(os.path.join(root, "Questions.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        idx = int(r["Index"])
        options = parse_mmvp_options(r["Options"])
        answer = "abcdefgh".index(r["Correct Answer"].strip().strip("()"))
        if not 0 <= answer < len(options):
            raise ValueError(f"MMVP item {idx}: cannot parse {r['Options']!r}")
        items.append(dict(id=str(idx), question=r["Question"].strip(), options=options,
                          answer=answer, group=(idx - 1) // 2,
                          image=os.path.join(root, "MMVP Images", f"{idx}.jpg")))
    return items


def load_aokvqa() -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceM4/A-OKVQA",
                           "data/validation-00000-of-00001-b2bd0de231b6326a.parquet",
                           repo_type="dataset")
    table = pq.read_table(path, columns=["question_id", "question", "choices",
                                         "correct_choice_idx", "image"])
    return [dict(id=r["question_id"], question=r["question"].strip(), options=r["choices"],
                 answer=int(r["correct_choice_idx"]), group=None, image=r["image"]["bytes"])
            for r in table.to_pylist()]


BENCHMARKS = {"mmvp": load_mmvp, "aokvqa": load_aokvqa}


def open_image(image) -> Image.Image:
    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    return Image.open(image).convert("RGB")
