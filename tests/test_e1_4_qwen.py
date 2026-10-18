"""E1.4 on Qwen2.5-VL: CPU checks with a tiny randomly initialised model and the real processor."""

import pytest
import torch
from PIL import Image

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
    pytest.skip("transformers without Qwen2.5-VL", allow_module_level=True)

from e1_4.qwen import QwenBackend, load_processor  # noqa: E402

SIZE = 112  # 4 x 4 visual tokens


@pytest.fixture(scope="module")
def backend():
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = "Qwen/Qwen2.5-VL-7B-Instruct"
    try:
        cfg = AutoConfig.from_pretrained(path)
        processor = load_processor(path, SIZE)
    except Exception:
        pytest.skip("Qwen2.5-VL config/processor not available")
    v, t = cfg.vision_config, cfg.text_config if hasattr(cfg, "text_config") else cfg
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.fullatt_block_indexes = 64, [1]
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 96, 2
    t.num_attention_heads, t.num_key_value_heads = 2, 1
    t.layer_types = ["full_attention"] * 2
    rope = getattr(t, "rope_parameters", None) or getattr(t, "rope_scaling", None)
    rope["mrope_section"] = [4, 6, 6]
    if not hasattr(cfg, "text_config"):
        cfg.hidden_size, cfg.num_hidden_layers = 64, 2
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    return QwenBackend(model, processor, SIZE, batch_size=3, device="cpu")


def _images(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (160, 120), (70 * i, 40, 200 - 50 * i)).save(p)
        paths.append(str(p))
    return paths


def test_states_shapes_and_question_independent_image_tokens(backend, tmp_path):
    paths = _images(tmp_path)
    s = backend.states("Is there a person in the image?", paths, [0, 2], grid=2)
    assert s["img_L0"].shape == (3, 4 * 64) and s["last_L2"].shape == (3, 64)
    # image tokens come before the question: their states must not depend on it
    other = backend.states("How many people are in the image?", paths, [0, 2], grid=2)
    assert abs(s["img_L2"].astype(float) - other["img_L2"].astype(float)).max() < 1e-2
    assert abs(s["last_L2"].astype(float) - other["last_L2"].astype(float)).max() > 1e-3
    feats = backend.visual_features(paths, grid=2)
    assert abs(feats["grid"].astype(float) - s["img_L0"].astype(float)).max() < 1e-2
    with pytest.raises(ValueError):
        backend.states("q", paths[:1], [5], grid=2)


def test_option_scores_match_an_unpadded_forward(backend, tmp_path):
    paths = _images(tmp_path, 2)
    items = [dict(id=f"x{i}", question="How many people are in the image?",
                  options=["0", "1", "3 or more"], answer=i, image=p) for i, p in enumerate(paths)]
    res = backend.score(items)
    assert [r["n_tokens"] for r in res][0][2] > res[0]["n_tokens"][0]   # longer option, more tokens
    # reference: one sequence at a time, full logits, no padding
    tok = backend.processor.tokenizer
    for it, r in zip(items, res):
        for j, option in enumerate(it["options"]):
            text = backend.prompt(it["question"]) + option + "<|im_end|>"
            batch = backend._batch([text], [backend._image(it["image"])])
            with torch.no_grad():
                logits = backend.model(**batch).logits[0]
            ids = batch["input_ids"][0]
            n = len(tok(option + "<|im_end|>", add_special_tokens=False)["input_ids"])
            lp = torch.log_softmax(logits[-n - 1:-1].float(), -1)
            ref = float(-lp.gather(1, ids[-n:, None]).sum())
            assert r["n_tokens"][j] == n
            assert abs(r["nll_image"][j] - ref) < 1e-3 * max(1.0, abs(ref)), (option, r["nll_image"][j], ref)
    assert res[0]["nll_blank"] == res[1]["nll_blank"]   # blank image: same for every item


def test_image_size_must_fit_the_token_grid(backend):
    with pytest.raises(ValueError):
        QwenBackend(backend.model, backend.processor, 100, 1, "cpu")
