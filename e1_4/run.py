"""E1.4 bottleneck probe: does MiniVLM report what its frozen encoder encodes?

    python -m e1_4.run --work_dir runs/e1_4 --runs runs/main_clip ... --names clip ...

1. Build image-level questions about one COCO category (presence, count,
   position) from the val2017 box annotations, keeping only images whose
   label is the same in every encoder's view (tasks.py). Class-balanced test
   sets; the remaining images train the probes.
2. For each trained MiniVLM run: extract the patch features its projector
   receives, train linear probes on them (probe.py), and score MiniVLM's own
   answers to the same questions on the same test images by option likelihood.
   MiniVLM is also scored on a few training images per class, which fit its
   calibrated readout (a per-option offset, analysis.py).
3. Probe the projector output and the frozen LLM's hidden states at a few
   layers on the same images (llm_probe.py), to find the stage where a
   property stops being linearly readable.
4. Compare: probe accuracy is what the encoder makes linearly available;
   MiniVLM's accuracy is what survives the projector and the frozen LLM.

Resumable: finished runs (results_<name>.json) and extracted features are
reused. Defaults come from the `probe` section of config.yaml.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

from minivlm.data import coco_annotation_path, download_coco_images

from .analysis import format_summary, summarize, write
from .llm_probe import llm_states
from .probe import LinearProbe, pool, subsample_per_class
from .tasks import LabelRules, build_tasks, processor_geometry


@torch.no_grad()
def extract_features(model, processor, images: list, image_dir: str, batch_size: int,
                     grid: int, device: str) -> dict:
    out = {"mean": [], "grid": []}
    for start in range(0, len(images), batch_size):
        chunk = images[start: start + batch_size]
        pil = [Image.open(os.path.join(image_dir, n)).convert("RGB") for n in chunk]
        pixels = processor(pil, return_tensors="pt")["pixel_values"].to(device)
        pooled = pool(model.encode_image(pixels), grid)
        for k in out:
            out[k].append(pooled[k].half().cpu().numpy())
        print(f"features: {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def _score(probe, x, n_classes: int) -> list:
    """Per test item: logit of class 1 minus class 0 (binary tasks), else []."""
    if n_classes != 2:
        return []
    z = probe.logits(x)
    return (z[:, 1] - z[:, 0]).tolist()


def probe_task(feats: dict, index: dict, meta: dict, train_sizes: list, seed: int,
               device: str) -> dict:
    """Probes for each pooling, with a learning curve. Test predictions are kept
    per item (also for the curve), so every comparison can be paired."""
    ytr = [it["label"] for it in meta["train"]]
    tr = [index[it["image"]] for it in meta["train"]]
    te = [index[it["image"]] for it in meta["test"]]
    answers = [it["label"] for it in meta["test"]]
    out = {}
    for kind, x in feats.items():
        xt = torch.from_numpy(x).float().to(device)
        probe = LinearProbe(meta["n_classes"], seed=seed).fit(xt[tr], ytr)
        curve, curve_pred = {}, {}
        for n in train_sizes:
            sub = subsample_per_class(ytr, n, seed)
            p = LinearProbe(meta["n_classes"], weight_decay=probe.weight_decay, seed=seed)
            pred = p.fit(xt[[tr[i] for i in sub]], [ytr[i] for i in sub]).predict(xt[te])
            key = str(n) if n > 0 else "all"
            curve[key] = float(np.mean([a == b for a, b in zip(pred, answers)]))
            curve_pred[key] = pred
        out[kind] = dict(test_pred=probe.predict(xt[te]),
                         test_score=_score(probe, xt[te], meta["n_classes"]),
                         weight_decay=probe.weight_decay,
                         val_accuracy={str(k): v for k, v in probe.val_accuracy.items()},
                         curve=curve, curve_pred=curve_pred)
    return out


def layer_probes(stages: dict, encoder_grid: np.ndarray, items: list, n_train: int,
                 n_classes: int, seed: int, device: str) -> dict:
    """Grid probe on the encoder features and a probe per LLM stage, all on the
    same images: items[:n_train] train, the rest test."""
    y = [it["label"] for it in items]
    ytr, answers = y[:n_train], y[n_train:]
    out = {}
    for name, x in {"encoder": encoder_grid, **stages}.items():
        xt = torch.from_numpy(x).float().to(device)
        probe = LinearProbe(n_classes, seed=seed).fit(xt[:n_train], ytr)
        out[name] = dict(test_pred=probe.predict(xt[n_train:]), weight_decay=probe.weight_decay)
    out["_answers"] = answers
    return out


def _fingerprint(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def main(argv=None):
    from minivlm.config import yaml_from_argv

    parent, y = yaml_from_argv()
    p = y["probe"]
    ap = argparse.ArgumentParser(parents=[parent], description=__doc__.split("\n")[0])
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="trained MiniVLM run directories")
    ap.add_argument("--names", nargs="+", help="labels for the runs (default: directory names)")
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--category", default=p["category"])
    ap.add_argument("--plural", default=p["plural"])
    ap.add_argument("--min_area", type=float, default=p["min_area"])
    ap.add_argument("--max_count", type=int, default=p["max_count"])
    ap.add_argument("--position_margin", type=float, default=p["position_margin"])
    ap.add_argument("--train_sizes", nargs="+", type=int, default=p["train_sizes"],
                    help="probe training images per class for the learning curve; 0 = all")
    ap.add_argument("--calib_per_class", type=int, default=p["calib_per_class"],
                    help="training images per class for MiniVLM's calibrated readout")
    ap.add_argument("--llm_layers", nargs="+", type=int, default=p["llm_layers"],
                    help="LLM layers to probe (0 = projector output, the LLM's input)")
    ap.add_argument("--layer_train_per_class", type=int, default=p["layer_train_per_class"],
                    help="probe training images per class for the layer probes")
    ap.add_argument("--grid", type=int, default=p["grid"])
    ap.add_argument("--batch_size", type=int, default=p["batch_size"])
    ap.add_argument("--bootstrap_samples", type=int, default=y["eval"]["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=p["seed"])
    args = ap.parse_args(argv)

    from transformers import AutoImageProcessor

    from minivlm.config import Config
    from minivlm.eval import load_run, score_options

    names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    os.makedirs(args.work_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Tasks, with labels that agree in every encoder's view
    configs = {}
    for name, run in zip(names, args.runs):
        with open(os.path.join(run, "config.json")) as f:
            configs[name] = Config(**json.load(f))
    geometries = [processor_geometry(AutoImageProcessor.from_pretrained(c.encoder_spec["model_id"]))
                  for c in configs.values()]
    with open(coco_annotation_path(args.data_dir, "instances_val2017.json")) as f:
        instances = json.load(f)
    rules = LabelRules(min_area=args.min_area, max_count=args.max_count,
                       position_margin=args.position_margin)
    tasks = build_tasks(instances, args.category, args.plural, geometries, rules,
                        p["test_per_class"], args.seed)
    with open(os.path.join(args.work_dir, "tasks.json"), "w") as f:
        json.dump(dict(category=args.category, rules=vars(rules), geometries=geometries,
                       tasks=tasks), f, indent=1)
    for task, meta in tasks.items():
        print(f"{task}: {meta['question']} available per class {meta['available']}, "
              f"test {meta['test_per_class']} per class, probe training {len(meta['train'])} images")

    image_dir = os.path.join(args.data_dir, "val2017")
    images = sorted({it["image"] for m in tasks.values() for it in m["train"] + m["test"]})
    failed = download_coco_images(images, image_dir, split="val2017")
    if failed:
        raise SystemExit(f"{len(failed)} val2017 images failed to download; re-run to retry")
    index = {n: i for i, n in enumerate(images)}
    calib = {t: [m["train"][i] for i in subsample_per_class(
                 [it["label"] for it in m["train"]], args.calib_per_class, args.seed)]
             for t, m in tasks.items()}
    layer_items = {t: [m["train"][i] for i in subsample_per_class(
                       [it["label"] for it in m["train"]], args.layer_train_per_class, args.seed)]
                   + m["test"] for t, m in tasks.items()}
    fingerprint = _fingerprint(tasks, args.train_sizes, args.calib_per_class, args.llm_layers,
                               args.layer_train_per_class, args.grid, args.seed, 2)

    # 2-3. Per encoder: features, probes, MiniVLM's answers, LLM-stage probes
    results = {}
    for name, run in zip(names, args.runs):
        path = os.path.join(args.work_dir, f"results_{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            if saved.get("fingerprint") == fingerprint:
                results[name] = saved
                continue
            print(f"{name}: tasks or settings changed since {path} was written; recomputing")
        cfg, tokenizer, processor, model = load_run(run, device)
        cache = os.path.join(args.work_dir, f"features_{name}_grid{args.grid}.npz")
        if os.path.exists(cache) and list(np.load(cache)["images"]) == images:
            data = np.load(cache)
            feats = {"mean": data["mean"], "grid": data["grid"]}
        else:
            feats = extract_features(model, processor, images, image_dir, args.batch_size,
                                     args.grid, device)
            np.savez(cache, images=np.array(images), **feats)
        out = dict(run=run, encoder=cfg.encoder, fingerprint=fingerprint, tasks={})
        for task, meta in tasks.items():
            print(f"{name} / {task}: probes", flush=True)
            probes = probe_task(feats, index, meta, args.train_sizes, args.seed, device)

            def as_items(subset, kind):
                return [dict(id=f"{task}:{kind}:{it['image']}", question=meta["question"],
                             options=meta["options"], answer=it["label"],
                             image=os.path.join(image_dir, it["image"])) for it in subset]

            vlm = score_options(model, tokenizer, processor, cfg, as_items(meta["test"], "test"),
                                args.batch_size, device, label=f"{name} / {task}: MiniVLM")
            vlm_calib = score_options(model, tokenizer, processor, cfg,
                                      as_items(calib[task], "calib"), args.batch_size, device,
                                      label=f"{name} / {task}: MiniVLM calibration images")
            li = layer_items[task]
            stages = llm_states(model, tokenizer, processor, cfg, meta["question"],
                                [it["image"] for it in li], image_dir, args.llm_layers,
                                args.grid, args.batch_size, device)
            print(f"{name} / {task}: layer probes", flush=True)
            layers = layer_probes(stages, feats["grid"][[index[it["image"]] for it in li]], li,
                                  len(li) - len(meta["test"]), meta["n_classes"], args.seed, device)
            out["tasks"][task] = dict(probe=probes, vlm=vlm, vlm_calib=vlm_calib, layers=layers)
        with open(path, "w") as f:
            json.dump(out, f)
        results[name] = out
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Compare
    summary = summarize(results, args.bootstrap_samples, args.seed)
    print(format_summary(summary, tasks))
    write(summary, tasks, args.work_dir)
    print(f"\nsaved summary.json, bottleneck_probe.png and information_by_stage.png in {args.work_dir}")


if __name__ == "__main__":
    main()
