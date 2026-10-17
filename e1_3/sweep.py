"""Run Qwen2.5-VL-7B through VLMEvalKit datasets at fixed visual-token budgets.

    python -m e1_3.sweep --work_dir runs/e1_3

For every budget and dataset: generate answers with VLMEvalKit's Qwen2.5-VL
wrapper and the dataset's own prompt, score them with the dataset's official
evaluator, and save per-question scores to
<work_dir>/tokens_<budget>/<dataset>/scores.json. Re-running resumes: finished
questions and finished evaluations are skipped. Defaults come from the
`resolution` section of config.yaml.
"""

import argparse
import json
import os
import time

from .subsets import sample_indices
from .tokens import budget_pixels, native_tokens, visual_tokens


def use_sdpa_without_flash_attn() -> bool:
    """VLMEvalKit loads Qwen2.5-VL with FlashAttention-2. Fall back to PyTorch's
    scaled-dot-product attention when flash-attn is not installed; the outputs
    are the same up to numerical noise, only slower."""
    try:
        import flash_attn  # noqa: F401
        return False
    except ImportError:
        pass
    import transformers

    for name in ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        original = cls.from_pretrained.__func__

        def from_pretrained(klass, *args, _original=original, **kwargs):
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "sdpa"
            return _original(klass, *args, **kwargs)

        cls.from_pretrained = classmethod(from_pretrained)
    return True


def load_subset(name: str, n: int, seed: int):
    from vlmeval.dataset import build_dataset

    ds = build_dataset(name)
    if ds is None:
        raise ValueError(f"VLMEvalKit could not build dataset {name}")
    keep = set(sample_indices(name, ds.data["index"], n, seed))
    ds.data = ds.data[ds.data["index"].astype(int).isin(keep)].reset_index(drop=True)
    return ds


def image_size(ds, row) -> tuple[int, int]:
    """(height, width) of the row's first image."""
    from PIL import Image

    paths = ds.dump_image(row)
    path = paths[0] if isinstance(paths, list) else paths
    with Image.open(path) as im:
        return im.height, im.width


def group_of(ds, name: str, record) -> str:
    if ds.TYPE == "MCQ":
        for col in ["l2-category", "category"]:
            if col in record and isinstance(record[col], str):
                return f"{name}: {record[col]}"
    return name


def extract_scores(ds, name: str, eval_file: str) -> dict:
    """{index: (score, group)} from the per-question files the evaluators write."""
    from vlmeval.smp.file import get_intermediate_file_path, load

    if ds.TYPE == "MCQ":
        rec = load(get_intermediate_file_path(eval_file, "_exact_matching_result"))
        scores = rec["hit"]
    else:
        rec = load(get_intermediate_file_path(eval_file, "_results"))
        scores = rec["eval_score"]
    return {str(int(i)): (float(s), group_of(ds, name, rec.iloc[k]))
            for k, (i, s) in enumerate(zip(rec["index"], scores))}


def check_against_official(official, items: dict, label: str):
    """The per-question scores we keep must average to the evaluator's own Overall."""
    try:
        rows = official if isinstance(official, list) else [official]
        overall = float(next(r["Overall"] for r in rows if "Overall" in r))
    except (StopIteration, KeyError, TypeError, ValueError):
        print(f"({label}: no Overall in the official result to cross-check)")
        return
    overall = overall / 100 if overall > 1 else overall  # MCQ reports fractions, VQA percent
    ours = sum(it["score"] for it in items.values()) / len(items)
    if abs(ours - overall) > 0.005:
        print(f"WARNING {label}: per-question scores average {ours:.4f}, "
              f"official Overall {overall:.4f}")


def check_resize_rule(height: int, width: int, budget: int):
    """Warn if our token arithmetic disagrees with the installed qwen_vl_utils."""
    try:
        from qwen_vl_utils import smart_resize
        h, w = smart_resize(height, width, factor=28, min_pixels=budget_pixels(budget),
                            max_pixels=budget_pixels(budget))
    except Exception as err:  # older/newer signature: skip the check
        print(f"(could not cross-check token counts with qwen_vl_utils: {err})")
        return
    ours = visual_tokens(height, width, budget_pixels(budget), budget_pixels(budget))
    if (h // 28) * (w // 28) != ours:
        print(f"WARNING: token count mismatch for a {height}x{width} image at budget {budget}: "
              f"qwen_vl_utils {(h // 28) * (w // 28)}, ours {ours}")


def run(args):
    os.environ.setdefault("LMUData", args.lmu_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    if use_sdpa_without_flash_attn():
        print("flash-attn not installed: using PyTorch SDPA attention")
    from vlmeval.smp.file import dump, get_pred_file_format
    from vlmeval.vlm import Qwen2VLChat

    datasets = {name: load_subset(name, args.n, args.seed) for name in args.datasets}
    sizes = {name: {str(int(r["index"])): image_size(ds, r) for _, r in ds.data.iterrows()}
             for name, ds in datasets.items()}
    for name, ds in datasets.items():
        print(f"{name}: {len(ds.data)} rows")

    model = None
    for budget in args.budgets:
        for name, ds in datasets.items():
            out = os.path.join(args.work_dir, f"tokens_{budget:05d}", name)
            scores_path = os.path.join(out, "scores.json")
            if os.path.exists(scores_path):
                continue
            os.makedirs(out, exist_ok=True)
            preds_path = os.path.join(out, "predictions.json")
            preds = json.load(open(preds_path)) if os.path.exists(preds_path) else {}
            todo = [r for _, r in ds.data.iterrows() if str(int(r["index"])) not in preds]

            if todo:
                if model is None:
                    model = Qwen2VLChat(model_path=args.model_path,
                                        min_pixels=budget_pixels(budget),
                                        max_pixels=budget_pixels(budget),
                                        max_new_tokens=args.max_new_tokens,
                                        use_custom_prompt=False)
                    h, w = next(iter(sizes[name].values()))
                    check_resize_rule(h, w, budget)
                # The wrapper passes these to qwen_vl_utils for every image.
                model.min_pixels = model.max_pixels = budget_pixels(budget)
                model.set_dump_image(ds.dump_image)
                for k, row in enumerate(todo):
                    if model.use_custom_prompt(name):
                        struct = model.build_prompt(row, dataset=name)
                    else:
                        struct = ds.build_prompt(row)
                    start = time.time()
                    pred = model.generate(message=struct, dataset=name)
                    preds[str(int(row["index"]))] = dict(prediction=pred,
                                                         seconds=time.time() - start)
                    if (k + 1) % 20 == 0 or k + 1 == len(todo):
                        with open(preds_path, "w") as f:
                            json.dump(preds, f)
                        print(f"tokens {budget} / {name}: {len(preds)}/{len(ds.data)}", flush=True)

            data = ds.data.drop(columns=["image"], errors="ignore").copy()
            data["prediction"] = [preds[str(int(i))]["prediction"] for i in data["index"]]
            eval_file = os.path.join(out, f"{name}.{get_pred_file_format()}")
            dump(data, eval_file)
            summary = ds.evaluate(eval_file, model="exact_matching", nproc=4)
            items = {}
            for idx, (score, group) in extract_scores(ds, name, eval_file).items():
                h, w = sizes[name][idx]
                items[idx] = dict(score=score, group=group,
                                  tokens=visual_tokens(h, w, budget_pixels(budget),
                                                       budget_pixels(budget)),
                                  native_tokens=native_tokens(h, w),
                                  seconds=preds[idx]["seconds"])
            official = (json.loads(summary.to_json(orient="records"))
                        if hasattr(summary, "to_json") else summary)
            check_against_official(official, items, f"tokens {budget} / {name}")
            result = dict(budget=budget, dataset=name, n_items=len(items), official=official,
                          items=items)
            with open(scores_path, "w") as f:
                json.dump(result, f, indent=1)
            print(f"tokens {budget} / {name}: official result {result['official']}", flush=True)


def parse_args(argv=None):
    from minivlm.config import yaml_from_argv

    parent, y = yaml_from_argv()
    r = y["resolution"]
    ap = argparse.ArgumentParser(parents=[parent], description=__doc__.split("\n")[0])
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--lmu_dir", default=r["lmu_dir"], help="where VLMEvalKit stores datasets")
    ap.add_argument("--model_path", default=r["model_path"])
    ap.add_argument("--datasets", nargs="+", default=r["datasets"])
    ap.add_argument("--budgets", nargs="+", type=int, default=r["budgets"],
                    help="visual tokens per image")
    ap.add_argument("--n", type=int, default=r["n_per_dataset"],
                    help="questions per dataset (MMBench: question groups); 0 = all")
    ap.add_argument("--max_new_tokens", type=int, default=r["max_new_tokens"])
    ap.add_argument("--seed", type=int, default=r["seed"])
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
