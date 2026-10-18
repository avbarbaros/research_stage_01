"""Where along MiniVLM does the information disappear?

For each task, the question is put to MiniVLM exactly as in the option scoring
(image, question, short-answer prompt, assistant turn opened) and the frozen
LLM's hidden states are read at a few layers:

    img_L{l}   the 256 image-token states at layer l, pooled to the same g×g
               grid as the encoder probe. Layer 0 is the LLM's input, i.e. the
               projector output.
    last_L{l}  the state at the last prompt position, the one whose output is
               the first answer token. It has seen the image and the question.

Linear probes on these, next to the encoder grid probe on the same images,
show the stage at which a property stops being linearly readable: encoder →
projector → LLM image tokens → answer position → MiniVLM's answer.

The image block comes before the question in the prompt and attention is
causal, so the image-token states do not depend on the question.
"""

import os

import numpy as np
import torch
from PIL import Image

from minivlm.data import Collator, build_prompt

from .probe import pool


def stage_names(layers: list) -> list:
    return [f"img_L{l}" for l in layers] + [f"last_L{l}" for l in layers]


@torch.no_grad()
def llm_states(model, tokenizer, processor, cfg, question: str, images: list, image_dir: str,
               layers: list, grid: int, batch_size: int, device: str) -> dict:
    """{stage: array [len(images), dim]} for the stages of `stage_names(layers)`."""
    from minivlm.eval import ANSWER_PROMPT, autocast

    n_layers = model.llm.config.num_hidden_layers
    bad = [l for l in layers if not 0 <= l <= n_layers]
    if bad:
        raise ValueError(f"layers {bad} outside 0..{n_layers}")
    prompt = torch.tensor(build_prompt(tokenizer, f"{question}\n{ANSWER_PROMPT}",
                                       cfg.n_image_tokens))
    collate = Collator(tokenizer.pad_token_id)
    out = {k: [] for k in stage_names(layers)}
    for start in range(0, len(images), batch_size):
        chunk = images[start: start + batch_size]
        seqs = [dict(input_ids=prompt, labels=prompt,
                     pixel_values=processor(Image.open(os.path.join(image_dir, n)).convert("RGB"),
                                            return_tensors="pt")["pixel_values"][0])
                for n in chunk]
        batch = {k: v.to(device) for k, v in collate(seqs).items()}
        with autocast(device):
            embeds = model.embed(batch["input_ids"], batch["pixel_values"])
            hidden = model.llm.base_model(inputs_embeds=embeds,
                                          attention_mask=batch["attention_mask"],
                                          output_hidden_states=True).hidden_states
        mask = batch["input_ids"] == model.image_token_id
        last = len(prompt) - 1  # same prompt for every image, so no padding
        for l in layers:
            h = hidden[l]
            img = h[mask].reshape(len(chunk), cfg.n_image_tokens, -1)  # patch order
            out[f"img_L{l}"].append(pool(img, grid)["grid"].half().cpu().numpy())
            out[f"last_L{l}"].append(h[:, last].float().half().cpu().numpy())
        print(f"LLM states: {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}
