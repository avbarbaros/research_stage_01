import pytest

from minivlm.metrics import (bootstrap, caption_metrics, first_sentence, is_finished,
                             paired_comparison)

pytest.importorskip("pycocoevalcap")

REFS = {
    "a.jpg": ["A dog runs on the beach.", "A brown dog running along the sand.",
              "A dog on a sandy beach.", "Dog running by the sea.", "A happy dog at the beach."],
    "b.jpg": ["A man rides a bicycle down the street.", "A cyclist on a city road.",
              "A person riding a bike in traffic.", "Man on a bicycle in the city.",
              "A bike rider on a busy street."],
    "c.jpg": ["Two cats sleep on a sofa.", "A pair of cats napping on a couch.",
              "Cats lying on a sofa.", "Two kittens asleep on the couch.", "Cats resting on a sofa."],
}


def records(captions, finished=None):
    return [dict(image=n, references=REFS[n], short=c,
                 **({} if finished is None else {"short_finished": finished[n]}))
            for n, c in captions.items()]


GOOD = {"a.jpg": "A dog running on the beach. It looks very happy and the sky is blue",
        "b.jpg": "A man riding a bicycle on a city street.",
        "c.jpg": "Two cats sleeping on a sofa. The room is"}
BAD = {"a.jpg": "A bowl of fruit on a table.", "b.jpg": "A plate of food.",
       "c.jpg": "A tall building at night."}


def test_first_sentence():
    assert first_sentence("A dog. It runs.") == "A dog."
    assert first_sentence("It costs 3.5 dollars! Really.") == "It costs 3.5 dollars!"
    assert first_sentence("No full stop here") == "No full stop here"


def test_is_finished_prefers_flag_and_falls_back_to_punctuation():
    assert is_finished({"short": "cut off mid", "short_finished": True})
    assert not is_finished({"short": "Ends well.", "short_finished": False})
    assert is_finished({"short": "Ends well."})
    assert not is_finished({"short": "cut off mid"})


def test_bootstrap_interval_contains_mean_and_is_reproducible():
    ci = bootstrap([0.1, 0.5, 0.9, 0.3], n_samples=2000, seed=1)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci == bootstrap([0.1, 0.5, 0.9, 0.3], n_samples=2000, seed=1)
    assert bootstrap([2.0, 2.0, 2.0], n_samples=100)["lo"] == 2.0


def test_caption_metrics_variants():
    m = caption_metrics(records(GOOD), n_bootstrap=200)
    assert m["n_images"] == 3
    assert m["finished_from"] == "punctuation heuristic"
    assert m["n_finished"] == 1  # only b.jpg ends with punctuation
    assert m["cider_finished"] is None  # needs at least two finished captions
    assert m["mean_words_first_sentence"] < m["mean_words_short"]

    flagged = caption_metrics(records(GOOD, {"a.jpg": True, "b.jpg": True, "c.jpg": False}),
                              n_bootstrap=200)
    assert flagged["finished_from"] == "generation"
    assert flagged["n_finished"] == 2 and flagged["cider_finished"]["n"] == 2

    assert m["cider"]["mean"] > caption_metrics(records(BAD), n_bootstrap=200)["cider"]["mean"]


def test_paired_comparison():
    result = paired_comparison({"good": records(GOOD), "bad": records(BAD),
                                "good2": records(GOOD)}, n_bootstrap=200)
    assert result["n_images"] == 3
    cider = result["metrics"]["cider"]
    assert cider["pairs"]["good - bad"]["mean"] > 0
    same = cider["pairs"]["good - good2"]
    assert same["mean"] == same["lo"] == same["hi"] == 0.0
    # first sentences are scored on the same three images
    assert result["metrics"]["cider_first_sentence"]["per_run"]["bad"]["n"] == 3


def test_paired_comparison_uses_common_images_only():
    partial = records({k: v for k, v in GOOD.items() if k != "c.jpg"})
    result = paired_comparison({"full": records(GOOD), "partial": partial}, n_bootstrap=100)
    assert result["n_images"] == 2


def test_bootstrap_ratio_is_token_weighted():
    from minivlm.metrics import bootstrap_ratio
    ci = bootstrap_ratio([10.0, 1.0], [10, 1], n_samples=500)
    assert ci["mean"] == 1.0 and ci["lo"] == ci["hi"] == 1.0
    ci = bootstrap_ratio([20.0, 1.0], [10, 1], n_samples=500)
    assert ci["mean"] == pytest.approx(21 / 11)


def test_likelihood_summary_and_paired_comparison():
    from minivlm.metrics import likelihood_summary, paired_likelihood_comparison
    a = {"x": dict(tokens=10, nll_correct=10.0, nll_shuffled=15.0),
         "y": dict(tokens=20, nll_correct=30.0, nll_shuffled=40.0)}
    b = {n: dict(r, nll_correct=r["nll_correct"] + r["tokens"] * 0.5) for n, r in a.items()}
    s = likelihood_summary(a, n_bootstrap=200)
    assert s["correct"]["mean"] == pytest.approx(40 / 30)
    assert s["gap"]["mean"] == pytest.approx(15 / 30)
    assert s["n_tokens"] == 30

    cmp = paired_likelihood_comparison({"a": a, "b": b}, n_bootstrap=200)
    diff = cmp["metrics"]["correct"]["pairs"]["a - b"]
    assert diff["mean"] == pytest.approx(-0.5) and diff["lo"] == pytest.approx(-0.5)
    assert cmp["metrics"]["gap"]["pairs"]["a - b"]["mean"] == pytest.approx(0.5)

    b["x"]["tokens"] = 11
    with pytest.raises(ValueError):
        paired_likelihood_comparison({"a": a, "b": b}, n_bootstrap=10)
