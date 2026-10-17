"""Evaluate a trained projector.

    python -m minivlm.eval grounding --data_dir /content/data --out_dir <run dir>
    python -m minivlm.eval caption   --data_dir /content/data --out_dir <run dir> --n 50 --cider
    python -m minivlm.eval rescore   --out_dir <run dir>

grounding: loss on held-out descriptions with the correct image, another
           sample's image, and a blank image. A grounded model has a clearly
           lower loss with the correct image.
caption:   captions for COCO val2017 images, optionally scored with CIDEr
           (whole caption, first sentence, finished captions; see metrics.py).
rescore:   recompute the metrics of an existing captions.json, no GPU needed.
"""

import argparse
import json
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import Config, yaml_from_argv
from .data import (Collator, LlavaDataset, build_prompt, download_coco_images,
                   load_coco_val_captions, subset_paths)
from .metrics import caption_metrics, format_metrics
from .model import MiniVLM

SHORT_PROMPT = "Describe the image in one short sentence."
DETAIL_PROMPT = "Describe the image in detail."


def load_run(out_dir: str, device: str):
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = Config(**json.load(f))
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm)
    image_processor = AutoImageProcessor.from_pretrained(cfg.encoder_spec["model_id"])
    model = MiniVLM.from_pretrained(cfg, tokenizer, device)
    model.load_projector(os.path.join(out_dir, "checkpoint.pt"))
    model.eval()
    return cfg, tokenizer, image_processor, model


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))


@torch.no_grad()
def grounding(args, device):
    cfg, tokenizer, image_processor, model = load_run(args.out_dir, device)
    _, holdout_path, image_dir = subset_paths(args.data_dir, args.subset)
    with open(holdout_path) as f:
        samples = json.load(f)
    dataset = LlavaDataset(samples, image_dir, tokenizer, image_processor,
                           cfg.n_image_tokens, cfg.max_text_tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=Collator(tokenizer.pad_token_id))

    totals = dict(correct=0.0, shuffled=0.0, blank=0.0)
    n_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if len(batch["input_ids"]) < 2:
            continue  # cannot shuffle a batch of one
        pixels = batch.pop("pixel_values")
        n = (batch["labels"][:, 1:] != -100).sum().item()
        variants = dict(correct=pixels, shuffled=pixels.roll(1, dims=0),
                        blank=torch.zeros_like(pixels))
        for name, pv in variants.items():
            with autocast(device):
                totals[name] += model(pixel_values=pv, **batch).loss.item() * n
        n_tokens += n

    result = {name: total / n_tokens for name, total in totals.items()}
    result["n_samples"] = len(samples)
    result["n_tokens"] = n_tokens
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out_dir, "grounding.json"), "w") as f:
        json.dump(result, f, indent=2)


@torch.no_grad()
def caption(args, device):
    cfg, tokenizer, image_processor, model = load_run(args.out_dir, device)
    refs = load_coco_val_captions(args.data_dir)
    names = sorted(refs)
    random.Random(args.seed).shuffle(names)
    names = names[: args.n]
    image_dir = os.path.join(args.data_dir, "val2017")
    failed = set(download_coco_images(names, image_dir, split="val2017"))
    names = [n for n in names if n not in failed]

    prompts = {"short": (SHORT_PROMPT, args.short_max_new_tokens),
               "detail": (DETAIL_PROMPT, args.max_new_tokens)}
    if args.short_only:
        prompts.pop("detail")
    records = {n: dict(image=n, references=refs[n]) for n in names}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for key, (question, max_new) in prompts.items():
        # Same prompt for every image, so every row has the same length: no padding.
        ids = torch.tensor(build_prompt(tokenizer, question, cfg.n_image_tokens), device=device)
        for start in range(0, len(names), args.batch_size):
            chunk = names[start: start + args.batch_size]
            images = [Image.open(os.path.join(image_dir, n)).convert("RGB") for n in chunk]
            pixels = image_processor(images, return_tensors="pt")["pixel_values"].to(device)
            input_ids = ids.unsqueeze(0).expand(len(chunk), -1)
            with autocast(device):
                out = model.generate(
                    input_ids, torch.ones_like(input_ids), pixels,
                    max_new_tokens=max_new, do_sample=False,
                    eos_token_id=im_end, pad_token_id=tokenizer.pad_token_id,
                )
            # A row that contains <|im_end|> stopped on its own; the rest hit the limit.
            finished = (out == im_end).any(dim=1).tolist()
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            for n, text, done in zip(chunk, texts, finished):
                records[n][key] = text.strip()
                records[n][f"{key}_finished"] = done
            print(f"{key}: {min(start + args.batch_size, len(names))}/{len(names)}", flush=True)

    result = dict(short_max_new_tokens=args.short_max_new_tokens,
                  captions=list(records.values()))
    if args.cider:
        add_metrics(result, args)

    with open(os.path.join(args.out_dir, "captions.json"), "w") as f:
        json.dump(result, f, indent=2)
    for r in list(records.values())[:5]:
        print(f"\n{r['image']}\n  short : {r['short']}\n  ref   : {r['references'][0]}")


def add_metrics(result: dict, args):
    metrics = caption_metrics(result["captions"], args.bootstrap_samples, args.seed)
    result["metrics"] = metrics
    result["cider_short"] = metrics["cider"]["mean"]  # kept for older readers
    print(format_metrics(metrics))


def rescore(args, device=None):
    path = os.path.join(args.out_dir, "captions.json")
    with open(path) as f:
        result = json.load(f)
    add_metrics(result, args)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"updated {path}")


def main():
    parent, y = yaml_from_argv()
    e = y["eval"]
    ap = argparse.ArgumentParser(parents=[parent])
    ap.add_argument("command", choices=["grounding", "caption", "rescore"])
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--subset", default=y["data"]["subset"])
    ap.add_argument("--out_dir", required=True, help="training run directory")
    ap.add_argument("--batch_size", type=int, default=e["batch_size"])
    ap.add_argument("--num_workers", type=int, default=e["num_workers"])
    ap.add_argument("--n", type=int, default=e["caption_n"], help="caption: number of val images")
    ap.add_argument("--short_max_new_tokens", type=int, default=e["short_max_new_tokens"])
    ap.add_argument("--max_new_tokens", type=int, default=e["max_new_tokens"],
                    help="token limit for the detailed-description prompt")
    ap.add_argument("--short_only", action="store_true")
    ap.add_argument("--cider", action=argparse.BooleanOptionalAction, default=e["cider"])
    ap.add_argument("--bootstrap_samples", type=int, default=e["bootstrap_samples"])
    ap.add_argument("--seed", type=int, default=e["seed"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"grounding": grounding, "caption": caption, "rescore": rescore}[args.command](args, device)


if __name__ == "__main__":
    main()
