"""Caption metrics: CIDEr variants with bootstrap confidence intervals.

CIDEr here uses a simple lowercase/alphanumeric tokeniser, not the PTB
tokeniser behind published COCO numbers, and its IDF comes from the references
of the images being scored. Scores are therefore comparable only between runs
scored on the same images.

Three variants, because the model writes LLaVA-style paragraphs where COCO
references are one short sentence:
    cider                 the whole "short" caption
    cider_first_sentence  only its first sentence: removes most of the length effect
    cider_finished        only captions that ended before the token limit
"""

import re
from itertools import combinations

import numpy as np

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def simple_tokens(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def first_sentence(text: str) -> str:
    """Text up to and including the first sentence-ending punctuation mark."""
    m = _SENTENCE_END.search(text)
    return text[: m.end()] if m else text


def is_finished(record: dict, key: str = "short") -> bool:
    """Whether generation ended on its own rather than at the token limit.

    Uses the flag written by `eval caption`; older captions.json files lack it,
    so fall back to "ends with sentence punctuation".
    """
    flag = record.get(f"{key}_finished")
    if flag is not None:
        return bool(flag)
    return record[key].rstrip().endswith((".", "!", "?", '"', "'"))


def cider_per_image(hyps: dict, refs: dict) -> dict:
    """{image: CIDEr} for one hypothesis per image. The corpus score is the mean."""
    from pycocoevalcap.cider.cider import Cider

    names = list(hyps)
    gts = {n: [simple_tokens(r) for r in refs[n]] for n in names}
    res = {n: [simple_tokens(hyps[n])] for n in names}
    _, scores = Cider().compute_score(gts, res)
    return dict(zip(names, (float(s) for s in scores)))


def bootstrap(values, n_samples: int = 5000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean of `values` with a percentile bootstrap interval, resampling items."""
    v = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_samples, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), n=len(v))


def bootstrap_ratio(nums, dens, n_samples: int = 5000, seed: int = 0,
                    level: float = 0.95) -> dict:
    """sum(nums) / sum(dens) with a percentile bootstrap interval, resampling items.

    For token-weighted means such as loss per token: items are images, nums
    their summed loss, dens their token counts.
    """
    num = np.asarray(list(nums), dtype=float)
    den = np.asarray(list(dens), dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_samples, len(num)))
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(ratios, [(1 - level) / 2, 1 - (1 - level) / 2])
    return dict(mean=float(num.sum() / den.sum()), lo=float(lo), hi=float(hi), n=len(num))


def _mean_words(texts) -> float:
    texts = list(texts)
    return float(np.mean([len(t.split()) for t in texts])) if texts else 0.0


def caption_metrics(records: list, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """Metrics for one run's caption records (the "captions" list of captions.json)."""
    refs = {r["image"]: r["references"] for r in records}
    short = {r["image"]: r["short"] for r in records}
    finished = [r["image"] for r in records if is_finished(r)]

    def ci(scores):
        return bootstrap(scores.values(), n_bootstrap, seed)

    metrics = dict(
        cider=ci(cider_per_image(short, refs)),
        cider_first_sentence=ci(cider_per_image(
            {n: first_sentence(t) for n, t in short.items()}, refs)),
        cider_finished=(ci(cider_per_image({n: short[n] for n in finished}, refs))
                        if len(finished) > 1 else None),
        n_images=len(records),
        n_finished=len(finished),
        finished_from=("generation" if all("short_finished" in r for r in records)
                       else "punctuation heuristic"),
        mean_words_short=_mean_words(short.values()),
        mean_words_first_sentence=_mean_words(first_sentence(t) for t in short.values()),
        mean_words_reference=_mean_words(t for r in records for t in r["references"]),
    )
    if all("detail" in r for r in records):
        metrics["mean_words_detail"] = _mean_words(r["detail"] for r in records)
    return metrics


def paired_comparison(runs: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """Compare runs captioned on the same images, pairing scores image by image.

    runs: {run name: caption records}. Differences are "first - second" in the
    order the runs are given. cider_finished uses only the images whose caption
    finished in every run, so all runs are scored on the same subset.
    """
    by_run = {name: {r["image"]: r for r in recs} for name, recs in runs.items()}
    common = sorted(set.intersection(*(set(r) for r in by_run.values())))
    finished_all = [n for n in common if all(is_finished(r[n]) for r in by_run.values())]
    refs = {n: next(iter(by_run.values()))[n]["references"] for n in common}

    variants = {
        "cider": (common, lambda t: t),
        "cider_first_sentence": (common, first_sentence),
        "cider_finished": (finished_all, lambda t: t),
    }
    result = dict(n_images=len(common), n_finished_in_all=len(finished_all), metrics={})
    for metric, (images, transform) in variants.items():
        if len(images) < 2:
            continue
        scores = {name: cider_per_image({n: transform(recs[n]["short"]) for n in images}, refs)
                  for name, recs in by_run.items()}
        result["metrics"][metric] = dict(
            per_run={name: bootstrap(s.values(), n_bootstrap, seed) for name, s in scores.items()},
            pairs={f"{a} - {b}": bootstrap([scores[a][n] - scores[b][n] for n in images],
                                           n_bootstrap, seed)
                   for a, b in combinations(scores, 2)},
        )
    return result


# Reference-caption likelihood. Each measure is a token-weighted (or caption-
# weighted) ratio sum(num) / sum(den) over images, computed from per-image sums
# written by `eval reference`. "end" is the closing <|im_end|> of each caption:
# its loss shows how ready the model is to stop after one short sentence.
# The end-token measures need files written after the split was added.
LIKELIHOOD_MEASURES = {
    "correct": ("loss per token, correct image",
                lambda r: (r["nll_correct"], r["tokens"])),
    "gap": ("grounding gap per token, shuffled - correct",
            lambda r: (r["nll_shuffled"] - r["nll_correct"], r["tokens"])),
    "content_correct": ("content tokens only: loss per token, correct image",
                        lambda r: (r["nll_correct"] - r["nll_end_correct"],
                                   r["tokens"] - r["end_tokens"])),
    "content_gap": ("content tokens only: grounding gap per token",
                    lambda r: ((r["nll_shuffled"] - r["nll_end_shuffled"])
                               - (r["nll_correct"] - r["nll_end_correct"]),
                               r["tokens"] - r["end_tokens"])),
    "end_correct": ("<|im_end|> only: loss per caption, correct image",
                    lambda r: (r["nll_end_correct"], r["end_tokens"])),
    "end_gap": ("<|im_end|> only: grounding gap per caption",
                lambda r: (r["nll_end_shuffled"] - r["nll_end_correct"], r["end_tokens"])),
    "end_share": ("<|im_end|> contribution to the loss per token, correct image",
                  lambda r: (r["nll_end_correct"], r["tokens"])),
}


def _available(per_image_dicts) -> list:
    rows = [r for p in per_image_dicts for r in p.values()]
    has_end = all("nll_end_correct" in r for r in rows)
    return [m for m in LIKELIHOOD_MEASURES if has_end or not m.startswith(("content", "end"))]


def likelihood_summary(per_image: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """All available measures for one run, from {image: per-image sums}."""
    rows = list(per_image.values())
    summary = {}
    for m in _available([per_image]):
        num, den = zip(*(LIKELIHOOD_MEASURES[m][1](r) for r in rows))
        summary[m] = bootstrap_ratio(num, den, n_bootstrap, seed)
    summary["n_images"] = len(rows)
    summary["n_tokens"] = int(sum(r["tokens"] for r in rows))
    return summary


def paired_likelihood_comparison(runs: dict, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """Compare reference-caption likelihood of runs, image by image.

    runs: {run name: per_image dict from reference_likelihood.json}. Differences
    are "first - second": a negative loss difference means the first run gives
    the references a lower loss; a positive gap difference means it is more
    grounded.
    """
    common = sorted(set.intersection(*(set(p) for p in runs.values())))
    if any(len({runs[r][n]["tokens"] for r in runs}) > 1 for n in common):
        raise ValueError("runs tokenised the references differently; are the LLMs the same?")

    result = dict(n_images=len(common),
                  n_tokens=int(sum(next(iter(runs.values()))[n]["tokens"] for n in common)),
                  metrics={})
    for m in _available(runs.values()):
        fn = LIKELIHOOD_MEASURES[m][1]
        num = {name: [fn(runs[name][n])[0] for n in common] for name in runs}
        den = [fn(next(iter(runs.values()))[n])[1] for n in common]
        result["metrics"][m] = dict(
            description=LIKELIHOOD_MEASURES[m][0],
            per_run={name: bootstrap_ratio(num[name], den, n_bootstrap, seed) for name in runs},
            pairs={f"{a} - {b}": bootstrap_ratio([x - y for x, y in zip(num[a], num[b])],
                                                 den, n_bootstrap, seed)
                   for a, b in combinations(runs, 2)},
        )
    return result


def format_likelihood(m: dict) -> str:
    lines = [f"{LIKELIHOOD_MEASURES[k][0]:62s} {format_ci(m[k])}"
             for k in LIKELIHOOD_MEASURES if k in m]
    lines.append(f"({m['n_images']} images, {m['n_tokens']} reference tokens; "
                 "95% bootstrap over images)")
    return "\n".join(lines)


def format_ci(ci: dict | None) -> str:
    if ci is None:
        return "n/a"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def format_metrics(m: dict) -> str:
    lines = [
        f"CIDEr, whole short caption    {format_ci(m['cider'])}   ({m['n_images']} images)",
        f"CIDEr, first sentence only    {format_ci(m['cider_first_sentence'])}",
        f"CIDEr, finished captions only {format_ci(m['cider_finished'])}   "
        f"({m['n_finished']}/{m['n_images']} finished, from {m['finished_from']})",
        f"mean words: short {m['mean_words_short']:.1f}, first sentence "
        f"{m['mean_words_first_sentence']:.1f}, reference {m['mean_words_reference']:.1f}"
        + (f", detail {m['mean_words_detail']:.1f}" if "mean_words_detail" in m else ""),
        "(95% bootstrap intervals over images; simple tokeniser, within-project comparison only)",
    ]
    return "\n".join(lines)
