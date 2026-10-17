"""Minimal VLM: frozen vision encoder -> trainable projector -> frozen LLM.

The vision-language interface, for the default config and batch size B:

    pixel_values             [B, 3, 224, 224]
    encoder hidden_states[-2] [B, 257, 1024]   1 CLS + 16x16 patches
    drop CLS                 [B, 256, 1024]
    projector                [B, 256, 1536]   LLM hidden size
    text embeds              [B, T, 1536]     T includes 256 <|image_pad|> slots
    after splice             [B, T, 1536]     image_pad rows replaced
    LLM last hidden state    [B, T, 1536]
    logits                   [S, vocab]       S = supervised positions in the batch
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

IMAGE_TOKEN = "<|image_pad|>"


class Projector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, kind: str = "mlp2", hidden: int = 0):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.Linear(vision_dim, llm_dim))
        elif kind == "mlp2":
            hidden = hidden or llm_dim
            self.net = nn.Sequential(
                nn.Linear(vision_dim, hidden), nn.GELU(), nn.Linear(hidden, llm_dim)
            )
        else:
            raise ValueError(f"unknown projector kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiniVLM(nn.Module):
    def __init__(self, vision_encoder: nn.Module, llm: nn.Module, image_token_id: int,
                 n_image_tokens: int, num_prefix_tokens: int = 1, feature_layer: int = -2,
                 projector: str = "mlp2", projector_hidden: int = 0):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.image_token_id = image_token_id
        self.n_image_tokens = n_image_tokens
        self.num_prefix_tokens = num_prefix_tokens
        self.feature_layer = feature_layer

        self.vision_dim = vision_encoder.config.hidden_size
        self.llm_dim = llm.config.hidden_size
        self.projector = Projector(self.vision_dim, self.llm_dim, projector, projector_hidden)

        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False
        self.vision_encoder.eval()
        self.llm.eval()

    @classmethod
    def from_pretrained(cls, cfg: Config, tokenizer, device: str = "cuda") -> "MiniVLM":
        import transformers

        spec = cfg.encoder_spec
        on_gpu = torch.device(device).type == "cuda"
        encoder_cls = getattr(transformers, spec["model_class"])
        vision_encoder = encoder_cls.from_pretrained(
            spec["model_id"], torch_dtype=torch.float16 if on_gpu else torch.float32
        )
        llm = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.llm, torch_dtype=torch.bfloat16 if on_gpu else torch.float32
        )
        if cfg.grad_checkpointing:
            llm.gradient_checkpointing_enable()

        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            raise ValueError(f"tokenizer for {cfg.llm} has no {IMAGE_TOKEN} token")

        model = cls(
            vision_encoder, llm, image_token_id,
            n_image_tokens=cfg.n_image_tokens,
            num_prefix_tokens=spec["num_prefix_tokens"],
            feature_layer=cfg.feature_layer,
            projector=cfg.projector,
            projector_hidden=cfg.projector_hidden,
        )
        return model.to(device)

    def train(self, mode: bool = True):
        # Only the projector ever leaves eval mode.
        super().train(False)
        self.projector.train(mode)
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor, trace: dict | None = None) -> torch.Tensor:
        """[B, 3, H, W] -> [B, n_image_tokens, vision_dim]"""
        dtype = next(self.vision_encoder.parameters()).dtype
        out = self.vision_encoder(pixel_values.to(dtype), output_hidden_states=True)
        hidden = out.hidden_states[self.feature_layer]
        feats = hidden[:, self.num_prefix_tokens:]
        if feats.shape[1:] != (self.n_image_tokens, self.vision_dim):
            raise ValueError(
                f"expected [B, {self.n_image_tokens}, {self.vision_dim}] patch features, "
                f"got {tuple(feats.shape)}"
            )
        if trace is not None:
            trace["pixel_values"] = tuple(pixel_values.shape)
            trace[f"encoder hidden_states[{self.feature_layer}]"] = tuple(hidden.shape)
            trace["patch features (prefix dropped)"] = tuple(feats.shape)
        return feats

    def embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
              trace: dict | None = None) -> torch.Tensor:
        """Text embeddings with the image-token slots overwritten by projected features."""
        feats = self.encode_image(pixel_values, trace)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.projector(feats.to(self.projector.net[0].weight.dtype))

        mask = input_ids == self.image_token_id
        if not torch.all(mask.sum(dim=1) == self.n_image_tokens):
            raise ValueError(
                f"every sample needs exactly {self.n_image_tokens} image tokens, "
                f"got {mask.sum(dim=1).tolist()}"
            )
        # masked_scatter fills in row-major order, so each sample's slots receive
        # that sample's features in patch order.
        inputs_embeds = text_embeds.masked_scatter(
            mask.unsqueeze(-1), image_embeds.to(text_embeds.dtype)
        )
        if trace is not None:
            trace["projector output"] = tuple(image_embeds.shape)
            trace["input_ids"] = tuple(input_ids.shape)
            trace["text embeds"] = tuple(text_embeds.shape)
            trace["inputs_embeds (after splice)"] = tuple(inputs_embeds.shape)
        return inputs_embeds

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, trace=None):
        """With labels: returns .loss and .logits for the supervised positions only.

        The vocabulary projection is the single largest tensor in the model
        ([B, T, 151936]), and most positions (system prompt, image slots,
        questions, padding) carry no loss, so the LM head is applied only where
        a label exists. The loss equals the usual shifted cross-entropy.
        """
        inputs_embeds = self.embed(input_ids, pixel_values, trace)
        if labels is None:
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            if trace is not None:
                trace["logits"] = tuple(out.logits.shape)
            return out

        logits, targets, keep = self._supervised_logits(inputs_embeds, attention_mask, labels, trace)
        loss = F.cross_entropy(logits.float(), targets[keep])
        return SimpleNamespace(loss=loss, logits=logits)

    def _supervised_logits(self, inputs_embeds, attention_mask, labels, trace=None):
        """Logits at the supervised positions: [S, vocab], plus shifted targets and mask."""
        hidden = self.llm.base_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state
        targets = labels[:, 1:]  # position t predicts token t+1
        keep = targets != -100
        logits = self.llm.get_output_embeddings()(hidden[:, :-1][keep])
        if trace is not None:
            trace["LLM last hidden state"] = tuple(hidden.shape)
            trace["labels"] = tuple(labels.shape)
            trace["logits (supervised positions only)"] = tuple(logits.shape)
        return logits, targets, keep

    @torch.no_grad()
    def sequence_nll(self, input_ids, attention_mask, pixel_values, labels):
        """Per-sample negative log-likelihood of the supervised tokens.

        Returns (nll [B], n_tokens [B], last_nll [B]): the summed loss, the
        number of supervised tokens, and the loss of the last supervised token
        alone (the closing <|im_end|> of a single answer). nll / n_tokens is
        that sample's mean loss.
        """
        inputs_embeds = self.embed(input_ids, pixel_values)
        logits, targets, keep = self._supervised_logits(inputs_embeds, attention_mask, labels)
        nll = F.cross_entropy(logits.float(), targets[keep], reduction="none")
        rows = keep.nonzero()[:, 0]  # same row-major order as the boolean indexing above
        sums = torch.zeros(len(labels), device=nll.device).index_add_(0, rows, nll)
        counts = keep.sum(dim=1)
        last = nll[(counts.cumsum(0) - 1).clamp(min=0)]
        return sums, counts, torch.where(counts > 0, last, torch.zeros_like(last))

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **gen_kwargs):
        """Returns only the newly generated token ids."""
        inputs_embeds = self.embed(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_projector(self, path: str):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(state.get("projector", state))
