"""CPU tests with tiny randomly initialised models. No pretrained weights."""

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel, Qwen2Config, Qwen2ForCausalLM

from minivlm.data import IGNORE, SYSTEM_PROMPT, Collator, build_prompt, encode_conversation
from minivlm.model import IMAGE_TOKEN, MiniVLM

N_IMG = 4  # 28 px / patch 14 -> 2x2 patches
VISION_DIM, LLM_DIM, VOCAB, IMAGE_ID = 32, 48, 100, 7


def tiny_vlm(**kw):
    vision = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=VISION_DIM, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, image_size=28, patch_size=14))
    llm = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=VOCAB, hidden_size=LLM_DIM, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2))
    return MiniVLM(vision, llm, image_token_id=IMAGE_ID, n_image_tokens=N_IMG, **kw)


def tiny_batch(B=3, T=12):
    input_ids = torch.randint(10, VOCAB, (B, T))
    input_ids[:, 2:2 + N_IMG] = IMAGE_ID
    labels = input_ids.clone()
    labels[:, :8] = IGNORE
    return dict(input_ids=input_ids, attention_mask=torch.ones(B, T, dtype=torch.long),
                pixel_values=torch.randn(B, 3, 28, 28), labels=labels)


def test_interface_shapes():
    model, batch, trace = tiny_vlm(), tiny_batch(), {}
    out = model(**batch, trace=trace)
    assert trace["encoder hidden_states[-2]"] == (3, N_IMG + 1, VISION_DIM)
    assert trace["patch features (prefix dropped)"] == (3, N_IMG, VISION_DIM)
    assert trace["projector output"] == (3, N_IMG, LLM_DIM)
    assert trace["inputs_embeds (after splice)"] == (3, 12, LLM_DIM)
    assert trace["logits"] == (3, 12, VOCAB)
    assert torch.isfinite(out.loss)


def test_only_projector_trains():
    model = tiny_vlm()
    model.train()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("projector.") for n in trainable)
    assert model.projector.training
    assert not model.llm.training and not model.vision_encoder.training

    model(**tiny_batch()).loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.projector.parameters())
    assert all(p.grad is None for p in model.llm.parameters())


def test_splice_places_each_samples_features_in_its_own_slots():
    model, batch = tiny_vlm(), tiny_batch()
    embeds = model.embed(batch["input_ids"], batch["pixel_values"])
    expected = model.projector(model.encode_image(batch["pixel_values"]))
    assert torch.allclose(embeds[:, 2:2 + N_IMG], expected, atol=1e-6)
    text = model.llm.get_input_embeddings()(batch["input_ids"])
    assert torch.equal(embeds[:, 2 + N_IMG:], text[:, 2 + N_IMG:])


def test_wrong_image_token_count_raises():
    model, batch = tiny_vlm(), tiny_batch()
    batch["input_ids"][0, 2] = 11
    with pytest.raises(ValueError):
        model(**batch)


def test_generate_and_projector_roundtrip(tmp_path):
    model, batch = tiny_vlm(projector="linear"), tiny_batch()
    out = model.generate(batch["input_ids"], batch["attention_mask"], batch["pixel_values"],
                         max_new_tokens=5, min_new_tokens=5, do_sample=False, pad_token_id=0)
    assert out.shape == (3, 5)

    path = str(tmp_path / "p.pt")
    model.save_projector(path)
    other = tiny_vlm(projector="linear")
    other.load_projector(path)
    for a, b in zip(model.projector.parameters(), other.projector.parameters()):
        assert torch.equal(a, b)


def test_collator_pads_and_masks():
    items = [dict(input_ids=torch.arange(5), labels=torch.arange(5),
                  pixel_values=torch.zeros(3, 2, 2)),
             dict(input_ids=torch.arange(3), labels=torch.arange(3),
                  pixel_values=torch.zeros(3, 2, 2))]
    batch = Collator(pad_token_id=99)(items)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["input_ids"][1].tolist() == [0, 1, 2, 99, 99]
    assert batch["labels"][1].tolist() == [0, 1, 2, IGNORE, IGNORE]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]


# --- tokenisation: needs the real Qwen tokenizer (a few MB, downloaded once) ---

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # offline
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


CONV = [
    {"from": "human", "value": "<image>\nWhat is the cat doing?"},
    {"from": "gpt", "value": "The cat is sleeping on a sofa."},
    {"from": "human", "value": "What colour is it?"},
    {"from": "gpt", "value": "It is grey."},
]


def test_tokenizer_has_image_token(tokenizer):
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    assert image_id is not None and image_id != tokenizer.unk_token_id
    assert tokenizer(IMAGE_TOKEN, add_special_tokens=False)["input_ids"] == [image_id]


def test_encode_conversation_matches_chat_template_and_masks_labels(tokenizer):
    n = 6
    ids, labels = encode_conversation(tokenizer, CONV, n)
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    assert len(ids) == len(labels)
    assert ids.count(image_id) == n

    # Same layout as the tokenizer's own chat template, apart from the image block.
    reference = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "<|vision_start|>" + IMAGE_TOKEN * n
                                    + "<|vision_end|>\nWhat is the cat doing?"},
        {"role": "assistant", "content": "The cat is sleeping on a sofa."},
        {"role": "user", "content": "What colour is it?"},
        {"role": "assistant", "content": "It is grey."},
    ], tokenize=False)
    assert tokenizer.decode(ids) == reference

    supervised = tokenizer.decode([t for t, l in zip(ids, labels) if l != IGNORE])
    assert supervised == "The cat is sleeping on a sofa.<|im_end|>It is grey.<|im_end|>"


def test_truncation_respects_budget(tokenizer):
    n, budget = 6, 40
    long_conv = [CONV[0], {"from": "gpt", "value": "word " * 500}] + CONV[2:]
    ids, labels = encode_conversation(tokenizer, long_conv, n, max_text_tokens=budget)
    assert len(ids) == budget + n
    assert any(l != IGNORE for l in labels)

    # Later turns that do not fit are dropped whole.
    ids, _ = encode_conversation(tokenizer, CONV, n, max_text_tokens=45)
    assert "grey" not in tokenizer.decode(ids)
    assert "sofa" in tokenizer.decode(ids)


def test_build_prompt_ends_at_assistant(tokenizer):
    ids = build_prompt(tokenizer, "Describe the image.", 6)
    assert tokenizer.decode(ids).endswith("<|im_start|>assistant\n")
