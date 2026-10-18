"""Data for the minimal VLM: LLaVA-Instruct subset, COCO images, tokenisation.

Prepare a subset from the command line:

    python -m minivlm.data --data_dir /content/data --subset main --n_total 50000
"""

import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .model import IMAGE_TOKEN

LLAVA_REPO = "liuhaotian/LLaVA-Instruct-150K"
COCO_URL = "http://images.cocodataset.org"
SYSTEM_PROMPT = "You are a helpful assistant."
IGNORE = -100


# --------------------------------------------------------------------------- #
# Subset construction and image download
# --------------------------------------------------------------------------- #

def _load_llava_json(filename: str) -> list:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LLAVA_REPO, filename, repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def build_subset(n_total: int = 50000, n_holdout: int = 500, seed: int = 0,
                 detail_fraction: float = 0.46) -> tuple[list, list]:
    """Detail descriptions plus conversations, with a holdout split by image.

    With the defaults this is all ~23k detail descriptions and ~27k conversations.
    Holdout samples are detail descriptions whose images never appear in train.
    """
    rng = random.Random(seed)
    detail = _load_llava_json("detail_23k.json")
    conversation = _load_llava_json("conversation_58k.json")
    rng.shuffle(detail)
    rng.shuffle(conversation)

    holdout = detail[:n_holdout]
    holdout_images = {s["image"] for s in holdout}
    detail = [s for s in detail[n_holdout:] if s["image"] not in holdout_images]
    conversation = [s for s in conversation if s["image"] not in holdout_images]

    n_detail = min(len(detail), round(n_total * detail_fraction))
    train = detail[:n_detail] + conversation[: n_total - n_detail]
    rng.shuffle(train)
    return train, holdout


def _download_one(url: str, path: str, retries: int = 4) -> bool:
    if os.path.exists(path):
        return True
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, path)
                return True
        except requests.RequestException:
            pass
    return False


def download_coco_images(file_names, image_dir: str, split: str = "train2017",
                         workers: int = 32) -> list:
    """Fetch only the listed COCO images. Returns the file names that failed."""
    os.makedirs(image_dir, exist_ok=True)
    file_names = sorted(set(file_names))

    def job(name):
        return _download_one(f"{COCO_URL}/{split}/{name}", os.path.join(image_dir, name))

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(job, file_names), total=len(file_names), desc=f"COCO {split}"))
    return [name for name, good in zip(file_names, ok) if not good]


# Annotation files kept from COCO's 241 MB annotation zip (downloaded once).
COCO_ANNOTATIONS = ("captions_val2017.json", "instances_val2017.json")


def coco_annotation_path(data_dir: str, name: str) -> str:
    """Path to a COCO val2017 annotation file, downloading the zip if needed."""
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
        with requests.get(f"{COCO_URL}/annotations/annotations_trainval2017.zip",
                          stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path) as z:
            for keep in COCO_ANNOTATIONS:
                target = os.path.join(data_dir, keep)
                if not os.path.exists(target):
                    with open(target, "wb") as f:
                        f.write(z.read(f"annotations/{keep}"))
        os.remove(zip_path)
    return path


def load_coco_val_captions(data_dir: str) -> dict:
    """{file_name: [reference captions]} for COCO val2017."""
    path = coco_annotation_path(data_dir, "captions_val2017.json")
    with open(path) as f:
        coco = json.load(f)
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    refs = {}
    for ann in coco["annotations"]:
        refs.setdefault(names[ann["image_id"]], []).append(ann["caption"].strip())
    return refs


# --------------------------------------------------------------------------- #
# Tokenisation (Qwen ChatML layout, written out by hand so labels are explicit)
# --------------------------------------------------------------------------- #
#
#   <|im_start|>system\n{SYSTEM}<|im_end|>\n
#   <|im_start|>user\n<|vision_start|><|image_pad|>*N<|vision_end|>\n{question}<|im_end|>\n
#   <|im_start|>assistant\n{answer}<|im_end|>\n
#   ... further user / assistant turns without the image block
#
# Loss is taken only on {answer}<|im_end|>.

def _ids(tokenizer, text: str) -> list:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def image_block(tokenizer, n_image_tokens: int) -> list:
    tok = tokenizer.convert_tokens_to_ids
    return ([tok("<|vision_start|>")] + [tok(IMAGE_TOKEN)] * n_image_tokens
            + [tok("<|vision_end|>")] + _ids(tokenizer, "\n"))


def _user_ids(tokenizer, question: str, n_image_tokens: int, first: bool) -> list:
    ids = _ids(tokenizer, "<|im_start|>user\n")
    if first:
        ids += image_block(tokenizer, n_image_tokens)
    return ids + _ids(tokenizer, question + "<|im_end|>\n<|im_start|>assistant\n")


def build_prompt(tokenizer, question: str, n_image_tokens: int) -> list:
    """Token ids for a single-turn prompt, ending where the assistant starts."""
    system = _ids(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    return system + _user_ids(tokenizer, question, n_image_tokens, first=True)


def encode_conversation(tokenizer, conversations: list, n_image_tokens: int,
                        max_text_tokens: int = 256) -> tuple[list, list]:
    """LLaVA-format turns -> (input_ids, labels).

    Whole turns are kept while they fit in max_text_tokens (image tokens not
    counted). If the first answer alone is too long it is cut at the budget.
    """
    input_ids = _ids(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    labels = [IGNORE] * len(input_ids)
    budget = max_text_tokens + n_image_tokens
    newline = _ids(tokenizer, "\n")

    turns = list(zip(conversations[0::2], conversations[1::2]))
    for i, (human, gpt) in enumerate(turns):
        question = human["value"].replace("<image>", "").strip()
        user = _user_ids(tokenizer, question, n_image_tokens, first=(i == 0))
        answer = _ids(tokenizer, gpt["value"].strip() + "<|im_end|>")

        room = budget - len(input_ids) - len(user)
        if len(answer) + len(newline) > room:
            if i > 0:
                break
            answer = answer[: max(room, 1)]
            input_ids += user + answer
            labels += [IGNORE] * len(user) + answer
            break
        input_ids += user + answer + newline
        labels += [IGNORE] * len(user) + answer + [IGNORE] * len(newline)
    return input_ids, labels


class LlavaDataset(Dataset):
    def __init__(self, samples: list, image_dir: str, tokenizer, image_processor,
                 n_image_tokens: int, max_text_tokens: int = 256):
        self.samples = samples
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.n_image_tokens = n_image_tokens
        self.max_text_tokens = max_text_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        image = Image.open(os.path.join(self.image_dir, sample["image"])).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        input_ids, labels = encode_conversation(
            self.tokenizer, sample["conversations"], self.n_image_tokens, self.max_text_tokens
        )
        return dict(
            input_ids=torch.tensor(input_ids),
            labels=torch.tensor(labels),
            pixel_values=pixel_values,
        )


class Collator:
    """Right-pads to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        T = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), T), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), T), IGNORE, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), T), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attention_mask[i, :n] = 1
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=torch.stack([b["pixel_values"] for b in batch]),
        )


def subset_paths(data_dir: str, subset: str) -> tuple[str, str, str]:
    """(train.json, holdout.json, train image dir)"""
    root = os.path.join(data_dir, subset)
    return (os.path.join(root, "train.json"), os.path.join(root, "holdout.json"),
            os.path.join(data_dir, "train2017"))


def main():
    from .config import yaml_from_argv

    parent, y = yaml_from_argv()
    d = y["data"]
    ap = argparse.ArgumentParser(parents=[parent],
                                 description="Build the LLaVA subset and download its images.")
    ap.add_argument("--data_dir", default=y["paths"]["data_dir"])
    ap.add_argument("--subset", default=d["subset"])
    ap.add_argument("--n_total", type=int, default=d["n_total"])
    ap.add_argument("--n_holdout", type=int, default=d["n_holdout"])
    ap.add_argument("--detail_fraction", type=float, default=d["detail_fraction"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--workers", type=int, default=d["download_workers"])
    args = ap.parse_args()

    train_path, holdout_path, image_dir = subset_paths(args.data_dir, args.subset)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train, holdout = build_subset(args.n_total, args.n_holdout, args.seed, args.detail_fraction)
    failed = set(download_coco_images(
        [s["image"] for s in train + holdout], image_dir, workers=args.workers
    ))
    if failed:
        print(f"{len(failed)} images failed to download; dropping their samples")
        train = [s for s in train if s["image"] not in failed]
        holdout = [s for s in holdout if s["image"] not in failed]

    with open(train_path, "w") as f:
        json.dump(train, f)
    with open(holdout_path, "w") as f:
        json.dump(holdout, f)
    print(f"train: {len(train)} samples -> {train_path}")
    print(f"holdout: {len(holdout)} samples -> {holdout_path}")
    print(f"images: {len({s['image'] for s in train + holdout})} files in {image_dir}")


if __name__ == "__main__":
    main()
