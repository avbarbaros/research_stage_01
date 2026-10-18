"""Image-level questions about one COCO category, labelled from the box annotations.

Three tasks, for a category such as "person":
    presence  Is there a person in the image?                   Yes / No
    count     How many people are in the image?                 0 … 4, "5 or more"
    position  Is the person on the left or the right side?      Left / Right (one person only)

Each encoder sees a different part of the image: CLIP resizes the short side to
224 and crops the centre square, DINOv2 resizes to 256 and crops a 224 centre
(a tighter square), SigLIP squashes the whole image to 224×224. A label is
computed inside each encoder's view, and an image is kept for a task only if
the label is unambiguous and identical in every view. So every encoder is
tested on exactly the same items with exactly the same answers.
"""

import random
from dataclasses import dataclass

TASK_NAMES = ("presence", "count", "position")


@dataclass
class LabelRules:
    min_area: float = 0.01          # objects smaller than this share of the view are not counted
    max_count: int = 5              # last count class is "max_count or more"
    position_margin: float = 0.1    # left if centre < 0.5 - margin, right if > 0.5 + margin
    min_visible: float = 0.5        # share of an object's box inside the view needed to count it


# --------------------------------------------------------------------------- #
# What part of the original image an encoder sees
# --------------------------------------------------------------------------- #

def _size_dict(value) -> dict | None:
    """Image-processor sizes come as dicts, SizeDict objects (newer transformers) or ints."""
    if value is None:
        return None
    if isinstance(value, int):
        return {"shortest_edge": value}
    keys = ("height", "width", "shortest_edge", "longest_edge")
    get = value.get if isinstance(value, dict) else (lambda k: getattr(value, k, None))
    return {k: get(k) for k in keys if get(k) is not None}


def processor_geometry(processor) -> dict:
    """Resize-and-crop rule of a Hugging Face image processor."""
    size = _size_dict(processor.size)
    crop = None
    if getattr(processor, "do_center_crop", False):
        crop = _size_dict(getattr(processor, "crop_size", None))
        if crop and "shortest_edge" in crop:  # an int crop size means a square
            crop = {"height": crop["shortest_edge"], "width": crop["shortest_edge"]}
    if "shortest_edge" in size:
        return dict(resize="shortest", edge=size["shortest_edge"], crop=crop)
    return dict(resize="fixed", height=size["height"], width=size["width"], crop=crop)


def view_box(geom: dict, width: int, height: int) -> tuple:
    """(x0, y0, x1, y1) of the original image that ends up in the encoder's input."""
    if geom["resize"] == "shortest":
        sx = sy = geom["edge"] / min(width, height)
    else:
        sx, sy = geom["width"] / width, geom["height"] / height
    if not geom["crop"]:
        return (0.0, 0.0, float(width), float(height))
    vw = min(width, geom["crop"]["width"] / sx)
    vh = min(height, geom["crop"]["height"] / sy)
    x0, y0 = (width - vw) / 2, (height - vh) / 2
    return (x0, y0, x0 + vw, y0 + vh)


def in_view(box, view) -> tuple:
    """For a COCO box [x, y, w, h]: (share of the box inside the view,
    visible area as a share of the view, horizontal centre of the visible part
    as a share of the view width)."""
    x, y, w, h = box
    vx0, vy0, vx1, vy1 = view
    ix0, iy0, ix1, iy1 = max(x, vx0), max(y, vy0), min(x + w, vx1), min(y + h, vy1)
    if ix1 <= ix0 or iy1 <= iy0 or w <= 0 or h <= 0:
        return 0.0, 0.0, None
    inter = (ix1 - ix0) * (iy1 - iy0)
    view_area = (vx1 - vx0) * (vy1 - vy0)
    return inter / (w * h), inter / view_area, ((ix0 + ix1) / 2 - vx0) / (vx1 - vx0)


# --------------------------------------------------------------------------- #
# Labels in one view, and across views
# --------------------------------------------------------------------------- #

def labels_in_view(boxes: list, view: tuple, rules: LabelRules) -> dict:
    """{task: class index or None (ambiguous)} for the objects of one category."""
    counted, centres, visible, ambiguous = 0, [], 0, False
    for box in boxes:
        frac, area, cx = in_view(box, view)
        if frac == 0:
            continue
        visible += 1
        clearly_in = frac >= rules.min_visible and area >= rules.min_area
        negligible = area < rules.min_area / 4
        if clearly_in:
            counted += 1
            centres.append(cx)
        elif not negligible:
            ambiguous = True  # cut by the view's edge, or of borderline size
    if visible == 0:
        presence, count = 1, 0          # "No", zero objects
    elif ambiguous or counted == 0:
        presence = count = None         # only tiny or half-visible objects: unclear
    else:
        presence, count = 0, min(counted, rules.max_count)
    position = None
    if visible == 1 and counted == 1 and not ambiguous:
        if centres[0] < 0.5 - rules.position_margin:
            position = 0
        elif centres[0] > 0.5 + rules.position_margin:
            position = 1
    return dict(presence=presence, count=count, position=position)


def consistent_labels(boxes: list, crowd: bool, width: int, height: int,
                      geometries: list, rules: LabelRules) -> dict:
    """{task: class index} for the tasks whose label is the same in every view."""
    if crowd:
        return {}
    per_view = [labels_in_view(boxes, view_box(g, width, height), rules) for g in geometries]
    out = {}
    for task in TASK_NAMES:
        values = {v[task] for v in per_view}
        if len(values) == 1 and None not in values:
            out[task] = values.pop()
    return out


# --------------------------------------------------------------------------- #
# Questions and splits
# --------------------------------------------------------------------------- #

def task_specs(name: str, plural: str, max_count: int) -> dict:
    return {
        "presence": dict(question=f"Is there a {name} in the image?", options=["Yes", "No"]),
        "count": dict(question=f"How many {plural} are in the image?",
                      options=[str(i) for i in range(max_count)] + [f"{max_count} or more"]),
        "position": dict(question=f"Is the {name} on the left or the right side of the image?",
                         options=["Left", "Right"]),
    }


def balanced_split(items: list, n_classes: int, test_per_class: int, seed: int = 0):
    """Test set with the same number of items per class (at most half of the
    rarest class); everything else is the probe's training set."""
    by_class = {c: [it for it in items if it["label"] == c] for c in range(n_classes)}
    rarest = min(len(v) for v in by_class.values())
    k = min(test_per_class, rarest // 2)
    rng = random.Random(seed)
    test, train = [], []
    for c in range(n_classes):
        pool = sorted(by_class[c], key=lambda it: it["image"])
        rng.shuffle(pool)
        test += pool[:k]
        train += pool[k:]
    return sorted(train, key=lambda it: it["image"]), sorted(test, key=lambda it: it["image"]), k


def build_tasks(instances: dict, category: str, plural: str, geometries: list,
                rules: LabelRules, test_per_class: dict, seed: int = 0) -> dict:
    """From COCO instances_*.json: {task: {question, options, train, test, ...}}."""
    cat_ids = [c["id"] for c in instances["categories"] if c["name"] == category]
    if not cat_ids:
        raise ValueError(f"no COCO category named {category!r}")
    cat = cat_ids[0]
    boxes, crowd = {}, set()
    for a in instances["annotations"]:
        if a["category_id"] != cat:
            continue
        if a.get("iscrowd"):
            crowd.add(a["image_id"])
        else:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])

    per_task = {t: [] for t in TASK_NAMES}
    for im in instances["images"]:
        labels = consistent_labels(boxes.get(im["id"], []), im["id"] in crowd,
                                   im["width"], im["height"], geometries, rules)
        for task, label in labels.items():
            per_task[task].append(dict(image=im["file_name"], label=label))

    specs = task_specs(category, plural, rules.max_count)
    out = {}
    for task, items in per_task.items():
        n_classes = len(specs[task]["options"])
        train, test, k = balanced_split(items, n_classes, test_per_class[task], seed)
        out[task] = dict(**specs[task], n_classes=n_classes, test_per_class=k,
                         available=[sum(it["label"] == c for it in items) for c in range(n_classes)],
                         train=train, test=test)
    return out
