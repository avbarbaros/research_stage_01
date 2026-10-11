"""Train the projector only.

    python -m minivlm.train --data_dir /content/data --out_dir /content/drive/MyDrive/e1_1/main

Re-running the same command with --resume continues from the last checkpoint.
"""

import argparse
import csv
from dataclasses import fields
import json
import math
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenising happens in DataLoader workers

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import ENCODERS, Config, yaml_from_argv
from .data import Collator, LlavaDataset, subset_paths
from .model import MiniVLM


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(path, model, optimizer, step, cfg):
    tmp = path + ".tmp"
    torch.save(dict(projector=model.projector.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, config=cfg.to_dict()), tmp)
    os.replace(tmp, path)


def parse_args():
    parent, y = yaml_from_argv()
    d, t = Config.from_yaml(y), y["train"]
    flag = argparse.BooleanOptionalAction
    ap = argparse.ArgumentParser(parents=[parent])
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--subset", default=y["data"]["subset"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder", default=d.encoder, choices=list(ENCODERS))
    ap.add_argument("--llm", default=d.llm)
    ap.add_argument("--feature_layer", type=int, default=d.feature_layer)
    ap.add_argument("--projector", default=d.projector, choices=["mlp2", "linear"])
    ap.add_argument("--projector_hidden", type=int, default=d.projector_hidden)
    ap.add_argument("--grad_checkpointing", action=flag, default=d.grad_checkpointing)
    ap.add_argument("--max_text_tokens", type=int, default=d.max_text_tokens)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--weight_decay", type=float, default=d.weight_decay)
    ap.add_argument("--warmup_ratio", type=float, default=d.warmup_ratio)
    ap.add_argument("--batch_size", type=int, default=d.batch_size)
    ap.add_argument("--accum", type=int, default=d.accum)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--max_grad_norm", type=float, default=d.max_grad_norm)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--max_samples", type=int, default=0, help="use only the first N samples")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimiser steps")
    ap.add_argument("--save_every", type=int, default=t["save_every"], help="optimiser steps")
    ap.add_argument("--log_every", type=int, default=t["log_every"], help="optimiser steps")
    ap.add_argument("--num_workers", type=int, default=t["num_workers"])
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    log_path = os.path.join(args.out_dir, "train_log.csv")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm)
    image_processor = AutoImageProcessor.from_pretrained(cfg.encoder_spec["model_id"])
    model = MiniVLM.from_pretrained(cfg, tokenizer, device)
    model.train()
    params = model.trainable_parameters()
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f}M (projector only)")

    train_path, _, image_dir = subset_paths(args.data_dir, args.subset)
    with open(train_path) as f:
        samples = json.load(f)
    if args.max_samples:
        samples = samples[: args.max_samples]
    dataset = LlavaDataset(samples, image_dir, tokenizer, image_processor,
                           cfg.n_image_tokens, cfg.max_text_tokens)
    collate = Collator(tokenizer.pad_token_id)

    batches_per_epoch = len(dataset) // cfg.batch_size
    steps_per_epoch = batches_per_epoch // cfg.accum
    total_steps = steps_per_epoch * cfg.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    print(f"{len(dataset)} samples, {total_steps} optimiser steps "
          f"(batch {cfg.batch_size} x accum {cfg.accum})")

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from step {step}")
    else:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr", "samples_per_s", "peak_mem_gb"])

    t0, seen, running = time.time(), 0, 0.0
    start_epoch = step // steps_per_epoch if steps_per_epoch else 0
    for epoch in range(start_epoch, cfg.epochs):
        # Deterministic per-epoch order so a resumed run skips exactly what was done.
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(dataset), generator=g).tolist()
        done_batches = (step - epoch * steps_per_epoch) * cfg.accum
        order = order[done_batches * cfg.batch_size: batches_per_epoch * cfg.batch_size]
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=order,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=(device == "cuda"), drop_last=True)

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            if step >= total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = model(**batch).loss
            (loss / cfg.accum).backward()
            running += loss.item() / cfg.accum
            seen += len(batch["input_ids"])

            if (i + 1) % cfg.accum:
                continue
            lr = lr_at(step, total_steps, warmup, cfg.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                n = args.log_every if step % args.log_every == 0 else step % args.log_every
                rate = seen / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                print(f"step {step}/{total_steps}  loss {running / n:.4f}  lr {lr:.2e}  "
                      f"{rate:.1f} samples/s  peak {mem:.1f} GB", flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{running / n:.5f}", f"{lr:.3e}",
                                            f"{rate:.2f}", f"{mem:.2f}"])
                t0, seen, running = time.time(), 0, 0.0
            if step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, step, cfg)

    save_checkpoint(ckpt_path, model, optimizer, step, cfg)
    print(f"done: {step} steps, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
