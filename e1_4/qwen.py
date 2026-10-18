"""E1.4 probes on Qwen2.5-VL: does a fully trained VLM with 2-D positions have the gap?

    python -m e1_4.qwen --work_dir <runs_dir>/e1_4_probe --data_dir data

Same tasks, test images, probes and readouts as the MiniVLM runs (run.py), so
every number is directly comparable. Reads tasks.json from the MiniVLM run;
run scripts/18_bottleneck_probe.py first.

Differences forced by the model:
- Each image is resized to a fixed square (default 448×448 px = 16×16 = 256
  visual tokens, like MiniVLM), so the tokens form a fixed grid that can be
  pooled the same way. Squashing to a square is exactly SigLIP's view, in which
  every label was checked (tasks.py), so the labels stay valid.
- "Encoder" here is the vision tower's output after Qwen's 2×2 patch merger,
  i.e. the visual tokens the LLM receives (Qwen's merger is its projector, and
  E1.4 found no loss at MiniVLM's projector). So the encoder stage and layer 0
  are the same features; their two probe scores should agree.
- Visual tokens carry M-RoPE positions (time, height, width), the 2-D
  positional design MiniVLM lacks.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from .analysis import format_summary, summarize, write
from .llm_probe import stage_names
from .probe import pool
from .run import _fingerprint, layer_probes, probe_task, shared_splits

ANSWER_PROMPT = "Answer the question using a single word or phrase."  # as in minivlm.eval


class QwenBackend:
    """Option scoring and hidden states for a Qwen2.5-VL model at a fixed image size."""

    def __init__(self, model, processor, image_size: int, batch_size: int, device: str):
        if image_size % 28:
            raise ValueError("image_size must be a multiple of 28 (one visual token = 28×28 px)")
        self.model, self.processor = model, processor
        self.size, self.batch_size, self.device = image_size, batch_size, device
        self.side = image_size // 28                       # visual tokens per row
        self.n_tokens = self.side ** 2
        self.image_token_id = model.config.image_token_id
        self.processor.tokenizer.padding_side = "left"      # answers end at the same position
        mean = getattr(processor.image_processor, "image_mean", [0.5, 0.5, 0.5])
        self.blank = Image.new("RGB", (image_size, image_size), tuple(int(255 * m) for m in mean))
        cfg = getattr(model.config, "text_config", model.config)
        self.n_layers = cfg.num_hidden_layers

    # ------------------------------------------------------------------ inputs
    def _image(self, path_or_image):
        im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
        return im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)

    def prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{question}\n{ANSWER_PROMPT}"}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _batch(self, texts: list, images: list) -> dict:
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        n = (enc["input_ids"] == self.image_token_id).sum(dim=1)
        if not torch.all(n == self.n_tokens):
            raise ValueError(f"expected {self.n_tokens} visual tokens per image, got {n.tolist()}; "
                             "check the processor's min_pixels / max_pixels")
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------ hidden states
    @torch.no_grad()
    def states(self, question: str, images: list, layers: list, grid: int, label: str = "") -> dict:
        """{stage: [N, dim]}: image tokens pooled to a grid and the answer position, per layer."""
        bad = [l for l in layers if not 0 <= l <= self.n_layers]
        if bad:
            raise ValueError(f"layers {bad} outside 0..{self.n_layers}")
        text = self.prompt(question)
        out = {k: [] for k in stage_names(layers)}
        for start in range(0, len(images), self.batch_size):
            chunk = [self._image(p) for p in images[start: start + self.batch_size]]
            batch = self._batch([text] * len(chunk), chunk)
            hidden = self.model(**batch, output_hidden_states=True).hidden_states
            mask = batch["input_ids"] == self.image_token_id
            for l in layers:
                h = hidden[l]
                img = h[mask].reshape(len(chunk), self.n_tokens, -1)   # row-major token grid
                out[f"img_L{l}"].append(pool(img, grid)["grid"].half().cpu().numpy())
                out[f"last_L{l}"].append(h[:, -1].float().half().cpu().numpy())
            print(f"{label}: {min(start + self.batch_size, len(images))}/{len(images)}", flush=True)
        return {k: np.concatenate(v) for k, v in out.items()}

    @torch.no_grad()
    def visual_features(self, images: list, grid: int, label: str = "") -> dict:
        """{"mean", "grid"} pooled visual tokens as the LLM receives them (layer 0)."""
        s = self.states("Describe the image.", images, [0], grid, label)
        # the grid pooling of layer 0; the mean is the average of the grid cells
        g = s["img_L0"].astype(np.float32)
        d = g.shape[1] // (grid * grid)
        return {"mean": g.reshape(len(g), grid * grid, d).mean(axis=1).astype(np.float16),
                "grid": s["img_L0"]}

    # ----------------------------------------------------------- option scoring
    def _nll(self, texts: list, images: list, n_targets: list) -> list:
        batch = self._batch(texts, images)
        k = max(n_targets) + 1
        try:
            logits = self.model(**batch, logits_to_keep=k).logits[:, -k:]
        except TypeError:  # older transformers without logits_to_keep
            logits = self.model(**batch).logits[:, -k:]
        ids = batch["input_ids"]
        out = []
        for i, n in enumerate(n_targets):
            # position p predicts token p + 1; the last n tokens are the answer
            lp = torch.log_softmax(logits[i, k - n - 1: k - 1].float(), dim=-1)
            out.append(float(-lp.gather(1, ids[i, -n:, None]).sum()))
        return out

    @torch.no_grad()
    def score(self, items: list, label: str = "") -> list:
        """Same contract and output as minivlm.eval.score_options."""
        tok = self.processor.tokenizer
        rows = []  # (item index, option index, text, n target tokens)
        for i, it in enumerate(items):
            p = self.prompt(it["question"])
            for j, option in enumerate(it["options"]):
                n = len(tok(option + "<|im_end|>", add_special_tokens=False)["input_ids"])
                rows.append((i, j, p + option + "<|im_end|>", n))
        nll = {"image": {}, "blank": {}}
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start: start + self.batch_size]
            texts, ns = [r[2] for r in chunk], [r[3] for r in chunk]
            for cond in nll:
                ims = [self._image(items[r[0]]["image"]) if cond == "image" else self.blank
                       for r in chunk]
                for r, v in zip(chunk, self._nll(texts, ims, ns)):
                    nll[cond][r[:2]] = v
            print(f"{label}: {min(start + self.batch_size, len(rows))}/{len(rows)} options", flush=True)
        n_tok = {r[:2]: r[3] for r in rows}
        return [dict(id=it["id"], answer=it["answer"], group=it.get("group"),
                     question=it["question"], options=it["options"],
                     n_tokens=[n_tok[(i, j)] for j in range(len(it["options"]))],
                     nll_image=[nll["image"][(i, j)] for j in range(len(it["options"]))],
                     nll_blank=[nll["blank"][(i, j)] for j in range(len(it["options"]))])
                for i, it in enumerate(items)]


def load_processor(model_path: str, image_size: int):
    """Processor that keeps every image at exactly image_size² pixels (min = max)."""
    from transformers import AutoProcessor

    pixels = image_size * image_size
    try:  # newer transformers: the pixel range is the processor's `size`
        return AutoProcessor.from_pretrained(
            model_path, size={"shortest_edge": pixels, "longest_edge": pixels})
    except (TypeError, ValueError, KeyError):
        return AutoProcessor.from_pretrained(model_path, min_pixels=pixels, max_pixels=pixels)


def load_qwen(model_path: str, image_size: int, device: str):
    import transformers
    from transformers import Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", **{key: dtype}).to(device).eval()
    return model, load_processor(model_path, image_size)


def evaluate(backend, name: str, tasks: dict, images: list, image_dir: str, splits: dict,
             args, fingerprint: str, device: str) -> dict:
    index = {n: i for i, n in enumerate(images)}
    cache = os.path.join(args.work_dir, f"features_{name}_{args.image_size}px_grid{args.grid}.npz")
    if os.path.exists(cache) and list(np.load(cache)["images"]) == images:
        data = np.load(cache)
        feats = {"mean": data["mean"], "grid": data["grid"]}
    else:
        feats = backend.visual_features([os.path.join(image_dir, n) for n in images], args.grid,
                                        label=f"{name}: visual tokens")
        np.savez(cache, images=np.array(images), **feats)
    out = dict(run=args.model_path, encoder=name, image_size=args.image_size,
               fingerprint=fingerprint, tasks={})
    for task, meta in tasks.items():
        print(f"{name} / {task}: probes", flush=True)
        probes = probe_task(feats, index, meta, args.train_sizes, args.seed, device)

        def as_items(subset, kind):
            return [dict(id=f"{task}:{kind}:{it['image']}", question=meta["question"],
                         options=meta["options"], answer=it["label"],
                         image=os.path.join(image_dir, it["image"])) for it in subset]

        vlm = backend.score(as_items(meta["test"], "test"), label=f"{name} / {task}: answers")
        vlm_calib = backend.score(as_items(splits["calib"][task], "calib"),
                                  label=f"{name} / {task}: calibration images")
        li = splits["layer_items"][task]
        stages = backend.states(meta["question"], [os.path.join(image_dir, it["image"]) for it in li],
                                args.llm_layers, args.grid, label=f"{name} / {task}: LLM states")
        layers = layer_probes(stages, feats["grid"][[index[it["image"]] for it in li]], li,
                              len(li) - len(meta["test"]), meta["n_classes"], args.seed, device)
        out["tasks"][task] = dict(probe=probes, vlm=vlm, vlm_calib=vlm_calib, layers=layers)
    return out


def main(argv=None):
    from minivlm.config import yaml_from_argv

    parent, y = yaml_from_argv()
    p, q = y["probe"], y["probe"]["qwen"]
    ap = argparse.ArgumentParser(parents=[parent], description=__doc__.split("\n")[0])
    ap.add_argument("--work_dir", required=True, help="the MiniVLM probe run (has tasks.json)")
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--model_path", default=q["model_path"])
    ap.add_argument("--name", default=q["name"])
    ap.add_argument("--image_size", type=int, default=q["image_size"])
    ap.add_argument("--llm_layers", nargs="+", type=int, default=q["llm_layers"])
    ap.add_argument("--batch_size", type=int, default=q["batch_size"])
    ap.add_argument("--compare", nargs="*", default=p["names"],
                    help="MiniVLM results to include in the combined summary")
    ap.add_argument("--train_sizes", nargs="+", type=int, default=p["train_sizes"])
    ap.add_argument("--calib_per_class", type=int, default=p["calib_per_class"])
    ap.add_argument("--layer_train_per_class", type=int, default=p["layer_train_per_class"])
    ap.add_argument("--grid", type=int, default=p["grid"])
    ap.add_argument("--bootstrap_samples", type=int, default=y["eval"]["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=p["seed"])
    args = ap.parse_args(argv)

    tasks_path = os.path.join(args.work_dir, "tasks.json")
    if not os.path.exists(tasks_path):
        raise SystemExit(f"{tasks_path} not found: run scripts/18_bottleneck_probe.py first")
    with open(tasks_path) as f:
        tasks = json.load(f)["tasks"]
    image_dir = os.path.join(args.data_dir, "val2017")
    images = sorted({it["image"] for m in tasks.values() for it in m["train"] + m["test"]})
    missing = [n for n in images if not os.path.exists(os.path.join(image_dir, n))]
    if missing:
        from minivlm.data import download_coco_images
        if download_coco_images(missing, image_dir, split="val2017"):
            raise SystemExit("some val2017 images failed to download; re-run to retry")
    splits = shared_splits(tasks, args.calib_per_class, args.layer_train_per_class, args.seed)
    fingerprint = _fingerprint(tasks, args.train_sizes, args.calib_per_class, args.llm_layers,
                               args.layer_train_per_class, args.grid, args.seed,
                               args.model_path, args.image_size, "qwen1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = os.path.join(args.work_dir, f"results_{args.name}.json")
    result = None
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        if result.get("fingerprint") != fingerprint:
            print(f"{path} was written with other tasks or settings; recomputing")
            result = None
    if result is None:
        model, processor = load_qwen(args.model_path, args.image_size, device)
        backend = QwenBackend(model, processor, args.image_size, args.batch_size, device)
        result = evaluate(backend, args.name, tasks, images, image_dir, splits, args,
                          fingerprint, device)
        with open(path, "w") as f:
            json.dump(result, f)

    results = {}
    for n in args.compare:
        rp = os.path.join(args.work_dir, f"results_{n}.json")
        if os.path.exists(rp):
            with open(rp) as f:
                results[n] = json.load(f)
        else:
            print(f"(no {rp}; {n} left out of the comparison)")
    results[args.name] = result
    summary = summarize(results, args.bootstrap_samples, args.seed)
    print(format_summary(summary, tasks))
    print(f"For {args.name}, 'encoder' / enc is the vision tower's output after the patch merger, "
          "i.e. the visual tokens the LLM receives (the same features as img L0).")
    write(summary, tasks, args.work_dir, suffix=f"_{args.name}")
    print(f"\nsaved summary_{args.name}.json, bottleneck_probe_{args.name}.png and "
          f"information_by_stage_{args.name}.png in {args.work_dir}")


if __name__ == "__main__":
    main()
