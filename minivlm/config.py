"""Configuration for the minimal VLM (roadmap experiment E1.1)."""

import argparse
import os
from dataclasses import asdict, dataclass, fields

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config.yaml")


def load_yaml(path: str | None = None) -> dict:
    import yaml

    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def yaml_from_argv() -> tuple[argparse.ArgumentParser, dict]:
    """Read --config from the command line early, so its values can be argparse defaults.

    Returns a parent parser carrying --config, and the loaded settings.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default=DEFAULT_CONFIG, help="settings file")
    known, _ = parent.parse_known_args()
    return parent, load_yaml(known.config)


# Vision encoders that can be swapped in (E1.2). All three give 16x16 = 256
# patch tokens at 224 px, so the LLM sees the same sequence length for each.
#   num_prefix_tokens: non-patch tokens at the start of the sequence (CLS) that
#   are dropped before projection. SigLIP has none.
ENCODERS = {
    "clip": dict(
        model_id="openai/clip-vit-large-patch14",
        model_class="CLIPVisionModel",
        num_prefix_tokens=1,
        n_image_tokens=256,
    ),
    "siglip": dict(
        model_id="google/siglip-so400m-patch14-224",
        model_class="SiglipVisionModel",
        num_prefix_tokens=0,
        n_image_tokens=256,
    ),
    "dinov2": dict(
        model_id="facebook/dinov2-large",
        model_class="Dinov2Model",
        num_prefix_tokens=1,
        n_image_tokens=256,
    ),
    # Small encoder for debugging the pipeline on CPU; not part of the experiments.
    "clip-b32": dict(
        model_id="openai/clip-vit-base-patch32",
        model_class="CLIPVisionModel",
        num_prefix_tokens=1,
        n_image_tokens=49,
    ),
}


@dataclass
class Config:
    """Model and optimisation settings of one run. Defaults mirror config.yaml."""

    # model
    encoder: str = "clip"
    llm: str = "Qwen/Qwen2.5-1.5B-Instruct"
    feature_layer: int = -2  # penultimate layer, as in LLaVA
    projector: str = "mlp2"  # "mlp2" or "linear"
    projector_hidden: int = 0  # 0 -> LLM hidden size; small values make a bottleneck
    grad_checkpointing: bool = False

    # data
    max_text_tokens: int = 256  # budget for everything except the image tokens

    # optimisation
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    batch_size: int = 16
    accum: int = 2
    epochs: int = 1
    max_grad_norm: float = 1.0
    seed: int = 0

    @classmethod
    def from_yaml(cls, settings: dict) -> "Config":
        """Flatten the model / train sections (and data.max_text_tokens) of config.yaml."""
        flat = {**settings.get("model", {}), **settings.get("train", {})}
        if "max_text_tokens" in settings.get("data", {}):
            flat["max_text_tokens"] = settings["data"]["max_text_tokens"]
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in flat.items() if k in names})

    @property
    def encoder_spec(self) -> dict:
        return ENCODERS[self.encoder]

    @property
    def n_image_tokens(self) -> int:
        return self.encoder_spec["n_image_tokens"]

    def to_dict(self) -> dict:
        return asdict(self)
