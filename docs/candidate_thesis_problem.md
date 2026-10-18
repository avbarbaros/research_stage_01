# Candidate thesis problem: visual layout is encoded but not read out

**Stage 1 outcome · research_stage_01**

This note states the thesis problem that Stage 1 points to, the evidence for it from E1.1–E1.4, and what has to be shown next before it can be committed to. Details of each experiment are in the [E1.1–E1.2 report](MiniVLM_E1.1_E1.2.md), the [E1.3 report](E1.3_resolution_study.md) and the [E1.4 report](E1.4_bottleneck_probe.md).

---

## 1 · The problem in one paragraph

In a LLaVA-style vision-language model, the vision encoder captures *where* things are in an image, and that information survives the projector and every layer of the language model inside the image tokens. But it never reaches the position from which the model produces its answer. What the answer can use behaves like an **unordered pool of the image tokens**: properties that do not depend on token order (is something present, how much of it is there) get through, while properties defined by layout (left or right, and by extension spatial relations) are lost. The gap between what the encoder encodes and what the model reports is therefore not mainly a problem of the encoder, the resolution or the projector, but of **routing layout from the image tokens to the answer**. A second, methodological problem hides it: the language model's preference for certain answer words can make information that *is* routed look absent in accuracy, so accuracy alone cannot tell the two failures apart.

**Candidate thesis question:** *Why does spatial layout that is present in a VLM's visual tokens fail to reach its answer, and what is the smallest change to the architecture or training that makes it reach the answer?*

---

## 2 · How Stage 1 led here

Stage 1 asked where visual detail is lost between the pixels and the answer. Each experiment tested one candidate location.

| Where detail could be lost | Experiment | Finding | Thesis-worthy? |
|---|---|---|---|
| **Input resize** | E1.3 (Qwen2.5-VL-7B, 64–1,600 tokens) | Lost for reading tasks: ChartQA 32% → 85%, TextVQA 52% → 85% from 64 to 1,024 tokens. Hardly for general perception: MMBench 75% → 78%. Accuracy levels off at each image's own resolution. | No: well understood, and native-resolution models already address it. |
| **Encoder choice** | E1.2, E1.4 | The three encoders differ in E1.2's format-sensitive measures, but on E1.4's controlled tasks they encode presence, count and position equally well (no clear difference). | No, for these properties. |
| **Projector** | E1.1 ablation, E1.4 | A 64- or 16-wide projector carries almost nothing (grounding gap −95% / −91%, A-OKVQA at the no-image level). At standard width it loses nothing measurable: probes on its output match the encoder within 1–3 points. A linear projector is as good as the MLP. | Only as a known failure mode, not at standard width. |
| **Answer-word preferences of the frozen LLM** | E1.2, E1.4 | Large. CLIP's "Is there a person?" accuracy is 52% raw, 93% with two fitted numbers; CLIP's short-caption lead in E1.2 is 70–86% stop-token preference. | As a measurement problem (§3.2). |
| **Routing inside the LLM** | E1.4 | Position: 89–96% in the image tokens at every layer, chance (46–60%) at the answer position at every layer, MiniVLM's AUC ≈ 50%. Count: computed in the LLM (86–90% at the answer position, above any encoder probe) but only partly expressed (58–70%). | **Yes.** |

The location that survives is the last one, and it is the only one where the information demonstrably exists and still fails to arrive.

---

## 3 · The evidence

### 3.1 Layout survives up to the image tokens and stops there (E1.4)

Linear-probe accuracy for "is the person on the left or the right?" (COCO, one-person images, chance 50%; three MiniVLM runs with CLIP, SigLIP and DINOv2 encoders):

| Stage | CLIP | SigLIP | DINOv2 |
|---|---|---|---|
| Encoder, 4×4 grid pooling | 95.0 | 94.4 | 96.2 |
| Encoder, mean pooling (layout removed) | 56.2 | 46.2 | 50.6 |
| Projector output | 93.1 | 91.9 | 93.8 |
| LLM image tokens, last layer | 96.2 | 93.8 | 92.5 |
| LLM answer position, best of layers 7–28 | 47.5 | 54.4 | 60.0 |
| MiniVLM's answer, calibrated | 46.9 | 53.1 | 46.2 |
| MiniVLM's answer, AUC | 48.0 | 51.1 | 50.3 |

The same probes on presence and count show what *does* reach the answer position: presence at 98–99% and count at 86–90% (better than any probe on the encoder, so the LLM's attention performs real aggregation). Both are order-independent; position is the one property that depends on which tokens hold the content, and it is the one that is lost.

### 3.2 Accuracy confounds missing information with answer bias (E1.2, E1.4)

| "Is there a person in the image?" | CLIP | SigLIP | DINOv2 |
|---|---|---|---|
| MiniVLM raw accuracy | 51.7% (98% "Yes") | 96.0% | 59.3% (91% "Yes") |
| MiniVLM ranking of the images (AUC) | 98.8% | 98.9% | 99.0% |
| Calibrated on 16 images per class | 92.7% | 95.0% | 95.3% |

Raw accuracy would put a 35–44-point information gap on CLIP and DINOv2 that does not exist; for position, the same raw numbers look identical but there the information really is missing. E1.2 found the same confound in generation (stop-token preference) and E1.4 showed it in option likelihood. Any study of the gap needs a measurement that separates **information present**, **information routed** and **answer bias**; the stage probes plus AUC and calibration in E1.4 are a first version of such a protocol.

### 3.3 Consistent with the fine-grained failures seen earlier (E1.2)

MiniVLM's MMVP pair accuracy is 3–10% (chance 25%): it gives the same answer to both images of a pair, so the question decides, not the image. Many MMVP pairs differ in orientation, direction or spatial arrangement. E1.4 offers a concrete explanation for that class of failure, which E1.2 could only describe.

### 3.4 What is *not* the cause

- **Not resolution for these tasks.** E1.3 shows general perception questions are answered almost as well at 224 px (MiniVLM's input) as at full resolution, and the encoder probes reach 94–96% on position at 224 px.
- **Not the encoder.** All three encoders, trained very differently (contrastive with text, sigmoid with text, self-supervised without text), carry position equally well.
- **Not projector capacity** at standard width (§2).

---

## 4 · Why the explanation is plausible

The language model receives each image token's place only through its one-dimensional position in the sequence. Nothing in its text pre-training says that "the first 8 of every 16 tokens are the left half of a picture", and projector-only training on captions and conversations rarely asks where something is, so the only training signal that could teach this mapping is almost absent. Attention from the answer position can then compute content summaries (does any token look like a person; how many do) but not which tokens they were. Fully trained models such as Qwen2-VL and Qwen2.5-VL add 2-D positional information to image tokens (M-RoPE) and train the language model on grounding data, which suggests the field has met the same problem; whether those measures actually close the gap at the answer position, rather than in the image tokens, is not known to us yet.

---

## 5 · Research questions and hypotheses

**RQ1 · Generality.** Does the gap exist beyond MiniVLM: in fully trained VLMs (Qwen2.5-VL-7B, InternVL3) and for richer spatial properties (above/below, relations between two objects, relative size, orientation)?

- *H1:* Fully trained models narrow but do not close the gap: image tokens keep layout, the answer position recovers only part of it, and the remaining gap grows with the complexity of the spatial relation.
- *Falsified if* Qwen2.5-VL's answer-position probe matches its image-token probe on all spatial tasks. The thesis would then shift to "what minimal training produces this routing" (RQ3), with MiniVLM as the failure case.

**RQ2 · Mechanism.** Where exactly is layout dropped? Does each image token's hidden state encode its own location, or only its content? Does attention from the answer position look at the right tokens but merge them without position, or not look at them at all?

- *H2:* Image-token states encode content but little of their own location, so any pooling from the answer position is position-blind. Adding location to the token states should be necessary and largely sufficient.

**RQ3 · Remedy.** Which change closes the gap at the lowest cost: (a) training data that asks for positions and relations, (b) 2-D positional information on the image tokens (learned 2-D embeddings added at the projector, or M-RoPE-style rotary positions), (c) LoRA on the language model, (d) combinations?

- *H3:* Positional information alone (b) is not enough without a training signal that uses it (a); (a) + (b) with a frozen LLM closes most of the gap; (c) helps further but is not required.

**RQ4 · Measurement.** Can the stage-probe protocol (encoder → projector → image tokens → answer position → calibrated answer) serve as a general diagnostic that separates missing information, failed routing and answer bias, and does it predict benchmark failures such as MMVP's?

---

## 6 · Next experiments, in order

| # | Experiment | Answers | Cost |
|---|---|---|---|
| 1 | Same E1.4 probes on **Qwen2.5-VL-7B** (image tokens and answer position per layer, calibrated answers) | RQ1, decides the framing | One run of the existing pipeline with a new model wrapper |
| 2 | **More spatial tasks** on COCO boxes: above/below, "is X left of Y" between two objects, relative size; a second object category | RQ1 | Task definitions only; same pipeline |
| 3 | **Self-location probe:** predict each image token's own grid cell from its hidden state, per layer | RQ2 | Small; reuses the LLM forward pass |
| 4 | **Attention analysis** from the answer position to the image tokens on position questions | RQ2 | Small |
| 5 | MiniVLM variants: **spatial QA in training**, **2-D positional embeddings** at the projector, both, and **LoRA**; re-run the probes | RQ3 | 4–5 training runs of ~80 min on an L4 |
| 6 | Re-score MMVP and A-OKVQA with **AUC / calibrated readout**; check whether MMVP failures concentrate on spatial pairs | RQ4, revisits E1.2 | Analysis only |

Experiments 1 and 3 are the decision point: if the gap appears in Qwen2.5-VL and image tokens do not know their own location, the problem and its mechanism are both established.

---

## 7 · Limits of the current evidence

- **One model family.** All routing evidence comes from MiniVLM: frozen Qwen2.5-1.5B, projector-only training, 50k samples. The strongest claim so far is about *projector-only alignment*, not VLMs in general (hence RQ1).
- **One task, one category, one seed.** Position was tested only as left/right of a single person (160 test images, intervals ±7–8 points).
- **Linear probes.** They show what a linear readout can use; a property absent to a probe could be present nonlinearly. The language model's own head is linear, which is why this is the relevant measure, but it is a choice.
- **Image-token probes are generous.** The 4×4 pooling gives the probe each token's location for free, so high scores there show that content is kept, not that the model represents layout (this is what experiment 3 tests).
- **Prior work may already cover parts of this.** Spatial-reasoning weaknesses of VLMs are widely reported; the contribution has to be the localisation of the loss inside the model and a fix grounded in it. The literature check in §8 comes before committing.

---

## 8 · Reading list to check against

These are the works this problem has to be positioned against. Check the details of each before citing.

- Tong et al., 2024, *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs* (MMVP; CLIP-blind pairs).
- Tong et al., 2024, *Cambrian-1* (vision-centric evaluation and encoder comparisons for MLLMs).
- Kamath, Hessel and Chang, 2023, *What's "up" with vision-language models?* (spatial prepositions in VLMs).
- Rahmanzadehgervi et al., 2024, *Vision language models are blind* (simple geometric and spatial tests).
- Wang et al., 2024, *Qwen2-VL* (M-RoPE: 2-D positions for image tokens).
- Zhao et al., 2021, *Calibrate Before Use* (answer-prior calibration of language models).
- Alain and Bengio, 2016, *Understanding intermediate layers using linear classifier probes*.

---

## 9 · Fallback framings

If experiment 1 shows that fully trained models do not have the gap:

- **Training-signal framing:** which minimal data and positional design make a frozen-LLM VLM route layout (RQ3), with the probe protocol showing *when* during training routing appears.
- **Diagnostic framing:** the stage-probe protocol (RQ4) as a method to localise failures in VLMs, with layout routing in projector-only models and answer bias as its two case studies.

If the gap is real but already explained in the literature, the probe protocol and the cheapest fix (RQ3) remain the contribution.
