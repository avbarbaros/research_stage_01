"""Evaluate a trained projector.

    python -m minivlm.eval grounding --data_dir /content/data --out_dir <run dir>
    python -m minivlm.eval caption   --data_dir /content/data --out_dir <run dir> --n 50 --cider

grounding: loss on held-out descriptions with the correct image, another
           sample's image, and a blank image. A grounded model has a clearly
           lower loss with the correct image.
caption:   captions for COCO val2017 images, optionally scored with CIDEr.
"""

import argparse
import json
import os
import random
import re

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import Config, yaml_from_argv
from .data import (Collator, LlavaDataset, build_prompt, download_coco_images,
                   load_coco_val_captions, subset_paths)
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


def _simple_tokens(text: str) -> str:
    # Not the PTB tokeniser used for published COCO numbers, so CIDEr from here
    # is only comparable to other runs of this script.
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


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

    prompts = {"short": (SHORT_PROMPT, 40), "detail": (DETAIL_PROMPT, args.max_new_tokens)}
    if args.short_only:
        prompts.pop("detail")
    records = {n: dict(image=n, references=refs[n]) for n in names}

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
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    pad_token_id=tokenizer.pad_token_id,
                )
            for n, text in zip(chunk, tokenizer.batch_decode(out, skip_special_tokens=True)):
                records[n][key] = text.strip()
            print(f"{key}: {min(start + args.batch_size, len(names))}/{len(names)}", flush=True)

    result = dict(captions=list(records.values()))
    if args.cider:
        from pycocoevalcap.cider.cider import Cider

        gts = {n: [_simple_tokens(r) for r in records[n]["references"]] for n in names}
        res = {n: [_simple_tokens(records[n]["short"])] for n in names}
        score, _ = Cider().compute_score(gts, res)
        result["cider_short"] = float(score)
        print(f"CIDEr (short prompt, {len(names)} images, simple tokeniser): {score:.3f}")

    with open(os.path.join(args.out_dir, "captions.json"), "w") as f:
        json.dump(result, f, indent=2)
    for r in list(records.values())[:5]:
        print(f"\n{r['image']}\n  short : {r['short']}\n  ref   : {r['references'][0]}")


def main():
    parent, y = yaml_from_argv()
    e = y["eval"]
    ap = argparse.ArgumentParser(parents=[parent])
    ap.add_argument("command", choices=["grounding", "caption"])
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--subset", default=y["data"]["subset"])
    ap.add_argument("--out_dir", required=True, help="training run directory")
    ap.add_argument("--batch_size", type=int, default=e["batch_size"])
    ap.add_argument("--num_workers", type=int, default=e["num_workers"])
    ap.add_argument("--n", type=int, default=e["caption_n"], help="caption: number of val images")
    ap.add_argument("--max_new_tokens", type=int, default=e["max_new_tokens"])
    ap.add_argument("--short_only", action="store_true")
    ap.add_argument("--cider", action=argparse.BooleanOptionalAction, default=e["cider"])
    ap.add_argument("--seed", type=int, default=e["seed"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"grounding": grounding, "caption": caption}[args.command](args, device)


if __name__ == "__main__":
    main()
