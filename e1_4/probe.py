"""Linear probes on frozen vision-encoder features.

Features are the encoder outputs MiniVLM's projector receives (hidden_states[-2],
prefix dropped: [B, 256, D]), pooled two ways:
    mean  average over all 256 patches                       -> D values
    grid  average over a g×g grid of patch blocks (4×4)       -> g·g·D values
The mean loses where things are; the grid keeps a coarse layout, which
position and counting need. A grid probe can represent anything the mean probe
can (the mean is the average of the grid cells), so it is the primary probe.

The probe is multinomial logistic regression on standardised features, with an
L2 penalty chosen by balanced accuracy (mean per-class recall) on a stratified
20% validation part of the training split, then refit on the whole training
split. Classes are weighted so each counts equally, matching the balanced test
sets; plain accuracy on the imbalanced validation part would reward predicting
the majority class.
"""

import math
import random

import torch
import torch.nn.functional as F

WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def pool(feats: torch.Tensor, grid: int = 4) -> dict:
    """[B, N, D] patch features (N a square number) -> {"mean": [B, D], "grid": [B, g*g*D]}"""
    b, n, d = feats.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"{n} patches do not form a square grid")
    x = feats.float().reshape(b, side, side, d).permute(0, 3, 1, 2)
    g = F.adaptive_avg_pool2d(x, grid).permute(0, 2, 3, 1).reshape(b, -1)
    return {"mean": feats.float().mean(dim=1), "grid": g}


def _class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    return len(y) / (n_classes * counts)


def _fit(x: torch.Tensor, y: torch.Tensor, n_classes: int, weight_decay: float,
         max_iter: int = 300) -> tuple:
    w = torch.zeros(x.shape[1], n_classes, device=x.device, requires_grad=True)
    b = torch.zeros(n_classes, device=x.device, requires_grad=True)
    weights = _class_weights(y, n_classes)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y, weight=weights) + weight_decay * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def balanced_accuracy(pred: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    recalls = [float((pred[y == c] == c).float().mean()) for c in range(n_classes) if (y == c).any()]
    return sum(recalls) / len(recalls)


def _stratified_holdout(y: list, share: float, seed: int) -> tuple:
    rng = random.Random(seed)
    train, val = [], []
    for c in sorted(set(y)):
        idx = [i for i, v in enumerate(y) if v == c]
        rng.shuffle(idx)
        k = max(1, round(share * len(idx))) if len(idx) > 1 else 0
        val += idx[:k]
        train += idx[k:]
    return sorted(train), sorted(val)


class LinearProbe:
    def __init__(self, n_classes: int, weight_decay: float | None = None, seed: int = 0):
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.seed = seed
        self.val_accuracy = {}

    def fit(self, x: torch.Tensor, y: list) -> "LinearProbe":
        x = x.float()
        self.mu = x.mean(dim=0)
        self.sd = x.std(dim=0).clamp(min=1e-6)
        xs = (x - self.mu) / self.sd
        yt = torch.tensor(y, device=x.device)
        if self.weight_decay is None:  # choose the penalty on held-out training items
            tr, va = _stratified_holdout(y, 0.2, self.seed)
            for wd in WEIGHT_DECAYS:
                w, b = _fit(xs[tr], yt[tr], self.n_classes, wd)
                pred = (xs[va] @ w + b).argmax(dim=1)
                self.val_accuracy[wd] = balanced_accuracy(pred, yt[va], self.n_classes)
            self.weight_decay = max(WEIGHT_DECAYS, key=lambda wd: (self.val_accuracy[wd], wd))
        self.w, self.b = _fit(xs, yt, self.n_classes, self.weight_decay)
        return self

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu) / self.sd @ self.w + self.b

    def predict(self, x: torch.Tensor) -> list:
        return self.logits(x).argmax(dim=1).tolist()


def subsample_per_class(y: list, per_class: int, seed: int = 0) -> list:
    """Indices of at most per_class items of each class (all if per_class <= 0)."""
    if per_class <= 0:
        return list(range(len(y)))
    rng = random.Random(seed)
    keep = []
    for c in sorted(set(y)):
        idx = [i for i, v in enumerate(y) if v == c]
        rng.shuffle(idx)
        keep += idx[:per_class]
    return sorted(keep)
