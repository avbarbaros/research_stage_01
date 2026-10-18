import random

import pytest
import torch

from e1_4.probe import LinearProbe, pool, subsample_per_class
from e1_4.tasks import (LabelRules, balanced_split, build_tasks, consistent_labels, in_view,
                        labels_in_view, view_box)

CLIP = dict(resize="shortest", edge=224, crop={"height": 224, "width": 224})
DINO = dict(resize="shortest", edge=256, crop={"height": 224, "width": 224})
SIGLIP = dict(resize="fixed", height=224, width=224, crop=None)
R = LabelRules()


def test_view_boxes_for_a_640x480_image():
    assert view_box(CLIP, 640, 480) == pytest.approx((80, 0, 560, 480))
    # DINOv2: scale 256/480, so the 224 crop covers 420 original pixels
    assert view_box(DINO, 640, 480) == pytest.approx((110, 30, 530, 450))
    assert view_box(SIGLIP, 640, 480) == (0, 0, 640, 480)


def test_in_view():
    frac, area, cx = in_view([0, 0, 100, 100], (0, 0, 200, 200))
    assert (frac, area, cx) == (1.0, 0.25, 0.25)
    frac, _, _ = in_view([150, 0, 100, 100], (0, 0, 200, 200))
    assert frac == 0.5
    assert in_view([300, 300, 10, 10], (0, 0, 200, 200)) == (0.0, 0.0, None)


def test_labels_in_view_cases():
    view = (0, 0, 100, 100)
    assert labels_in_view([], view, R) == dict(presence=1, count=0, position=None)
    one_left = labels_in_view([[5, 20, 20, 40]], view, R)
    assert one_left == dict(presence=0, count=1, position=0)
    centred = labels_in_view([[40, 20, 20, 40]], view, R)
    assert centred["position"] is None and centred["count"] == 1
    six = labels_in_view([[i * 15, 10, 12, 30] for i in range(6)], view, R)
    assert six["count"] == 3 and six["position"] is None              # "3 or more"
    cut = labels_in_view([[90, 10, 40, 40]], view, R)                  # 25% inside the view
    assert cut == dict(presence=None, count=None, position=None)
    tiny = labels_in_view([[10, 10, 4, 4]], view, R)                   # 0.16% of the view
    assert tiny["presence"] is None
    tiny_and_big = labels_in_view([[10, 10, 1, 1], [60, 20, 30, 40]], view, R)
    assert tiny_and_big["count"] == 1 and tiny_and_big["position"] is None  # 2 visible


def test_labels_must_agree_across_views():
    # a person at the far left of a 640x480 image: visible to SigLIP only
    labels = consistent_labels([[0, 100, 60, 200]], False, 640, 480, [CLIP, DINO, SIGLIP], R)
    assert "presence" not in labels and "count" not in labels
    # a person well inside every view, right of centre
    labels = consistent_labels([[360, 100, 100, 250]], False, 640, 480, [CLIP, DINO, SIGLIP], R)
    assert labels == dict(presence=0, count=1, position=1)
    assert consistent_labels([[360, 100, 100, 250]], True, 640, 480, [CLIP], R) == {}


def test_balanced_split_and_build_tasks():
    items = [dict(image=f"{i}.jpg", label=i % 3) for i in range(30)] + \
            [dict(image=f"x{i}.jpg", label=0) for i in range(20)]
    train, test, k = balanced_split(items, 3, test_per_class=100, seed=0)
    assert k == 5 and len(test) == 15 and len(train) == 35
    assert [sum(it["label"] == c for it in test) for c in range(3)] == [5, 5, 5]
    assert not {it["image"] for it in train} & {it["image"] for it in test}

    rng = random.Random(0)
    images, anns = [], []
    for i in range(200):
        images.append(dict(id=i, file_name=f"{i}.jpg", width=640, height=480))
        for _ in range(rng.randint(0, 2)):
            anns.append(dict(image_id=i, category_id=1, iscrowd=0,
                             bbox=[rng.choice([130, 400]), 100, 80, 200]))
    inst = dict(images=images, annotations=anns, categories=[dict(id=1, name="person")])
    tasks = build_tasks(inst, "person", "people", [CLIP, DINO, SIGLIP], R,
                        dict(presence=20, count=10, position=10), seed=0)
    assert tasks["count"]["options"] == ["0", "1", "2", "3 or more"]
    assert tasks["presence"]["test_per_class"] == 20
    assert tasks["position"]["question"].startswith("Is the person")


def test_pool_shapes_and_mean_consistency():
    feats = torch.randn(2, 256, 8)
    p = pool(feats, grid=4)
    assert p["mean"].shape == (2, 8) and p["grid"].shape == (2, 128)
    # the mean of the grid cells equals the overall mean
    assert torch.allclose(p["grid"].reshape(2, 16, 8).mean(dim=1), p["mean"], atol=1e-5)


def test_linear_probe_learns_a_separable_task():
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(3, 20, generator=g) * 3
    y = [i % 3 for i in range(300)]
    x = centres[y] + torch.randn(300, 20, generator=g)
    probe = LinearProbe(3, seed=0).fit(x[:200], y[:200])
    acc = sum(p == t for p, t in zip(probe.predict(x[200:]), y[200:])) / 100
    assert acc > 0.9 and probe.weight_decay in probe.val_accuracy
    sub = subsample_per_class(y[:200], 5, seed=0)
    assert len(sub) == 15 and len(subsample_per_class(y, 0)) == 300


def test_processor_geometry_reads_dicts_sizedicts_and_ints():
    from types import SimpleNamespace
    from e1_4.tasks import processor_geometry
    as_dict = SimpleNamespace(size={"shortest_edge": 224}, do_center_crop=True,
                              crop_size={"height": 224, "width": 224})
    as_obj = SimpleNamespace(size=SimpleNamespace(shortest_edge=256, height=None, width=None),
                             do_center_crop=True, crop_size=SimpleNamespace(height=224, width=224))
    as_int = SimpleNamespace(size=224, do_center_crop=True, crop_size=224)
    fixed = SimpleNamespace(size={"height": 224, "width": 224}, do_center_crop=False)
    assert processor_geometry(as_dict) == CLIP
    assert processor_geometry(as_obj) == DINO
    assert processor_geometry(as_int) == CLIP
    assert processor_geometry(fixed) == SIGLIP


def test_real_processor_geometries():
    pytest.importorskip("transformers")
    from transformers import AutoImageProcessor
    from e1_4.tasks import processor_geometry
    try:
        got = [processor_geometry(AutoImageProcessor.from_pretrained(m)) for m in
               ["openai/clip-vit-large-patch14", "facebook/dinov2-large",
                "google/siglip-so400m-patch14-224"]]
    except OSError as err:  # offline
        pytest.skip(str(err))
    assert got == [CLIP, DINO, SIGLIP]


def test_probe_chooses_penalty_by_balanced_accuracy():
    from e1_4.probe import balanced_accuracy
    y = torch.tensor([0] * 8 + [1] * 2)
    assert balanced_accuracy(torch.zeros(10, dtype=torch.long), y, 2) == 0.5  # majority guess
    assert balanced_accuracy(y.clone(), y, 2) == 1.0


def test_option_bias_removes_a_fixed_preference():
    from e1_4.analysis import auc, calibrated_predictions, fit_option_bias
    rng = random.Random(0)
    items = []
    for i in range(200):
        a = i % 2
        # option 1 always costs 3 nats more, but the image moves it by 1 nat
        nll = [2.0 + rng.gauss(0, 0.3), 5.0 - (2.0 if a == 1 else 0.0) + rng.gauss(0, 0.3)]
        items.append(dict(nll_image=nll, answer=a))
    raw = [int(min(range(2), key=it["nll_image"].__getitem__) == it["answer"]) for it in items]
    assert sum(raw) / len(raw) == 0.5                      # always option 0
    t, off = fit_option_bias(items[:32])
    pred = calibrated_predictions(items[32:], t, off)
    acc = sum(p == it["answer"] for p, it in zip(pred, items[32:])) / len(pred)
    assert acc > 0.9
    score = [it["nll_image"][0] - it["nll_image"][1] for it in items]
    assert auc(score, [it["answer"] for it in items]) > 0.95
    assert auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_llm_states_shapes_and_projector_stage():
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from types import SimpleNamespace
    from PIL import Image
    from e1_4.llm_probe import llm_states, stage_names
    from minivlm.model import MiniVLM
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        pytest.skip("tokenizer not available")
    torch.manual_seed(0)
    vision = CLIPVisionModel(CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                              num_attention_heads=2, image_size=56, patch_size=14))
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=48, intermediate_size=96,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2))
    model = MiniVLM(vision, llm, tok.convert_tokens_to_ids("<|image_pad|>"), n_image_tokens=16).eval()
    proc = CLIPImageProcessor(size={"shortest_edge": 56}, crop_size={"height": 56, "width": 56})
    cfg = SimpleNamespace(n_image_tokens=16, max_text_tokens=256)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            Image.new("RGB", (80, 60), (60 * i, 0, 0)).save(os.path.join(d, f"{i}.jpg"))
        out = llm_states(model, tok, proc, cfg, "Is there a person?", ["0.jpg", "1.jpg", "2.jpg"],
                         d, [0, 2], 2, 2, "cpu")
        assert list(out) == stage_names([0, 2])
        assert out["img_L0"].shape == (3, 4 * 48) and out["last_L2"].shape == (3, 48)
        # layer 0 image tokens are the projector output, pooled to the grid
        pix = proc(Image.open(os.path.join(d, "1.jpg")).convert("RGB"), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            proj = model.projector(model.encode_image(pix))
        assert torch.allclose(pool(proj, 2)["grid"].half().float(),
                              torch.from_numpy(out["img_L0"][1]).float(), atol=1e-2)
        with pytest.raises(ValueError):
            llm_states(model, tok, proc, cfg, "q", ["0.jpg"], d, [5], 2, 2, "cpu")
