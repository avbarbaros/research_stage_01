"""Print every tensor shape at the vision-language interface for one real batch.

Uses the main-run model and the smoke subset (run 03_smoke_test.py first).
"""

import argparse
import json

from _common import CFG, DATA

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from minivlm import ENCODERS, Config, MiniVLM
from minivlm.data import Collator, LlavaDataset, subset_paths

cfg = Config.from_yaml(CFG)
ap = argparse.ArgumentParser()
ap.add_argument("--config", help="settings file (default: config.yaml)")
ap.add_argument("--encoder", default=cfg.encoder, choices=list(ENCODERS))
ap.add_argument("--llm", default=cfg.llm)
ap.add_argument("--subset", default=CFG["smoke"]["subset"])
ap.add_argument("--batch_size", type=int, default=4)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.encoder, cfg.llm = args.encoder, args.llm
tok = AutoTokenizer.from_pretrained(cfg.llm)
proc = AutoImageProcessor.from_pretrained(cfg.encoder_spec["model_id"])
model = MiniVLM.from_pretrained(cfg, tok, device)

train_json, _, image_dir = subset_paths(DATA, args.subset)
with open(train_json) as f:
    samples = json.load(f)
ds = LlavaDataset(samples, image_dir, tok, proc, cfg.n_image_tokens, cfg.max_text_tokens)
batch = Collator(tok.pad_token_id)([ds[i] for i in range(args.batch_size)])
batch = {k: v.to(device) for k, v in batch.items()}

trace = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
    model(**batch, trace=trace)
print()
for name, shape in trace.items():
    print(f"{name:36s} {list(shape)}")

n = cfg.n_image_tokens
ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
text = tok.decode(ids).replace("<|image_pad|>" * n, f"<|image_pad|>x{n}").replace(tok.pad_token, "")
print("\nOne training sequence (image slots collapsed):\n" + text)
print("\nSupervised part:\n" + tok.decode([t for t, l in zip(ids, labels) if l != -100])[:300])
