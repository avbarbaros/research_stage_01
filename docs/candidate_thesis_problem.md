# Candidate thesis problem: what makes visual layout reach a VLM's answer?

**Stage 1 outcome · research_stage_01**

This note states the thesis problem that Stage 1 points to, the evidence for it from E1.1–E1.4 including the Qwen2.5-VL-7B reference run, and what has to be shown before it can be committed to. Details of each experiment are in the [E1.1–E1.2 report](MiniVLM_E1.1_E1.2.md), the [E1.3 report](E1.3_resolution_study.md) and the [E1.4 report](E1.4_bottleneck_probe.md).

**Revision note.** An earlier version of this note proposed that VLMs in general fail to route layout to their answer. The Qwen2.5-VL-7B run in E1.4 (§7 of that report) falsified this for left/right position, exactly as the earlier note's RQ1 condition specified: Qwen's answer position matches its image tokens. The problem is reframed accordingly, from *whether* layout reaches the answer to *what makes it reach the answer*.

---

## 1 · The problem in one paragraph

In a LLaVA-style vision-language model, the vision encoder captures *where* things are, and that information survives the projector and every layer of the language model inside the image tokens. Whether it then reaches the position from which the model answers depends on how the model was built and trained. In MiniVLM (frozen text LLM with 1-D positions, only the projector trained on captions and conversations) it never does: what the answer can use behaves like an **unordered pool of the image tokens**, so presence and count get through and left/right does not. In Qwen2.5-VL-7B (jointly trained, 2-D positions for image tokens, grounding data), the *same* position information in the visual tokens is carried to the answer position in the middle layers and the model answers left/right perfectly. Between the two lies a **routing step inside the language model** that projector-only alignment does not create and full training does. Which ingredient creates it, and how, is not known from our results. A second, methodological problem runs through both models: answer-word preferences make information that *is* routed look absent in raw accuracy, so measuring this needs readouts that separate information from bias.

**Candidate thesis question:** *Which ingredient of VLM design and training — 2-D positional information for image tokens, adaptation of the language model, or spatial supervision — makes visual layout reach the answer position, and how is that routing implemented inside the language model?*

---

## 2 · How Stage 1 led here

Stage 1 asked where visual detail is lost between the pixels and the answer. Each experiment tested one candidate location.

| Where detail could be lost | Experiment | Finding | Thesis-worthy? |
|---|---|---|---|
| **Input resize** | E1.3 (Qwen2.5-VL-7B, 64–1,600 tokens) | Lost for reading tasks: ChartQA 32% → 85%, TextVQA 52% → 85% from 64 to 1,024 tokens. Hardly for general perception: MMBench 75% → 78%. Accuracy levels off at each image's own resolution. | No: well understood, and native-resolution models already address it. |
| **Encoder choice** | E1.2, E1.4 | The three encoders differ in E1.2's format-sensitive measures, but on E1.4's controlled tasks they encode presence, count and position equally well (no clear difference). | No, for these properties. |
| **Projector** | E1.1 ablation, E1.4 | A 64- or 16-wide projector carries almost nothing (grounding gap −95% / −91%, A-OKVQA at the no-image level). At standard width it loses nothing measurable: probes on its output match the encoder within 1–3 points. A linear projector is as good as the MLP. | Only as a known failure mode, not at standard width. |
| **Answer-word preferences** | E1.2, E1.4 | Large in MiniVLM (CLIP presence 52% raw → 93% calibrated; CLIP's short-caption lead in E1.2 is 70–86% stop-token preference) and present in Qwen (count 73% raw → 94.5% calibrated). | As a measurement problem (§3.3). |
| **Routing inside the LLM** | E1.4, with Qwen reference | MiniVLM: position at 89–96% in the image tokens at every layer, at chance at the answer position. Qwen2.5-VL-7B: same position information in its visual tokens, 100% at the answer position from layer 21. | **Yes:** the information exists in both models, and only one routes it. |

The last row is the only place where information demonstrably exists and still fails to arrive in one model while arriving in another — a contrast that can be studied.

---

## 3 · The evidence

### 3.1 Layout stops at the image tokens in MiniVLM (E1.4)

Linear-probe accuracy for "is the person on the left or the right?" (COCO val2017, one-person images, 160 test images, chance 50%):

| Stage | CLIP | SigLIP | DINOv2 |
|---|---|---|---|
| Encoder, 4×4 grid pooling | 95.0 | 94.4 | 96.2 |
| Encoder, mean pooling (layout removed) | 56.2 | 46.2 | 50.6 |
| Projector output | 93.1 | 91.9 | 93.8 |
| LLM image tokens, last layer | 96.2 | 93.8 | 92.5 |
| LLM answer position, best of layers 7–28 | 47.5 | 54.4 | 60.0 |
| MiniVLM's answer, calibrated | 46.9 | 53.1 | 46.2 |
| MiniVLM's answer, AUC | 48.0 | 51.1 | 50.3 |

The same probes show what *does* reach MiniVLM's answer position: presence at 98–99% and count at 86–90%, both order-independent. Position is the property that depends on which tokens hold the content, and it is the one lost.

### 3.2 A fully trained model routes the same information (E1.4, §7)

Same tasks, images and probes on Qwen2.5-VL-7B (images resized to 448×448 = 256 visual tokens, as in MiniVLM):

| Position | MiniVLM (3 encoders) | Qwen2.5-VL-7B |
|---|---|---|
| Visual tokens entering the LLM | 92–96 | 95 |
| Image tokens, last layer | 92–96 | 93 |
| Answer position, L7 / L14 / L21 / L28 | 46–60 at every layer | 64 / 92 / 100 / 100 |
| Answer: raw / calibrated / AUC | ≈50 / 46–53 / ≈50 | 100 / 99 / 100 |

- **Same information in, different routing.** The visual tokens carry position equally well in both models; the difference is entirely in the transfer to the answer position.
- **Routing happens in the middle layers** of Qwen's language model (between layers 7 and 21).
- **Presence and count** also reach Qwen's answer fully once bias is removed (presence 98%, count 94.5% calibrated vs 96.5% at the answer position).

### 3.3 Accuracy confounds missing information with answer bias (E1.2, E1.4)

| | CLIP presence | DINOv2 presence | Qwen count |
|---|---|---|---|
| Raw accuracy | 51.7% (98% "Yes") | 59.3% (91% "Yes") | 73.0% (never "3 or more") |
| Ranking of the images (AUC) or answer-position probe | AUC 98.8% | AUC 99.0% | probe 96.5% |
| Calibrated on 16 images per class | 92.7% | 95.3% | 94.5% |

Raw accuracy understates what the model can report by 21–41 points in these cases, while for MiniVLM's position the information really is missing — and raw numbers alone cannot tell the two apart. Any study of routing needs a measurement that separates **information present**, **information routed** and **answer bias**. E1.4's stage probes plus AUC and calibrated readouts are a first version of such a protocol, and it worked on both a 1.5B frozen-LLM model and a 7B fully trained one.

### 3.4 Consistent with earlier failures (E1.2)

MiniVLM's MMVP pair accuracy is 3–10% (chance 25%): it gives the same answer to both images of a pair, so the question decides, not the image. Many MMVP pairs differ in orientation, direction or spatial arrangement; the routing failure offers a concrete explanation for that class of error.

### 3.5 What is *not* the cause

- **Not resolution for these tasks.** E1.3 shows general perception questions are answered almost as well at 224 px as at full resolution, and the encoder probes reach 94–96% on position at 224 px.
- **Not the encoder.** Three very differently trained encoders carry position equally well, and so do Qwen's visual tokens.
- **Not projector capacity** at standard width (§2).

---

## 4 · Candidate explanations

MiniVLM's language model receives each image token's place only through its 1-D position in the sequence. Nothing in its text pre-training relates "the first 8 of every 16 tokens" to the left half of a picture, and projector-only training on captions and conversations rarely asks where something is. Attention from the answer position can then compute content summaries (does any token look like a person; how many do) but not which tokens they were.

Qwen2.5-VL differs in at least four ways, each of which could supply the missing routing:

1. **2-D positional information** for image tokens (M-RoPE), which makes a token's row and column explicit to attention.
2. **An adapted language model.** Its LLM was trained on multimodal data, so its attention can learn to read the image block; MiniVLM's LLM is frozen.
3. **Spatial supervision.** Its training data includes grounding and spatial questions; MiniVLM's 50k LLaVA samples rarely ask for positions.
4. **Scale.** 7B instead of 1.5B parameters.

Our results establish that some combination of these works; they cannot separate them. Qwen has also very likely seen COCO images in training, so part of its performance may be memorisation.

---

## 5 · Research questions and hypotheses

**RQ1 · Ingredient.** Which of the four differences in §4 creates the routing? Starting from MiniVLM, where routing is absent, which minimal change — 2-D position embeddings on the image tokens, spatial questions in training, LoRA on the language model, or combinations — makes position readable at the answer position and in the answer?

- *H1:* Positional information alone is not enough without a training signal that uses it, and spatial supervision alone is not enough without a way for tokens to express their location; the two together, even with a frozen LLM, recover most of the routing. Adapting the LLM helps further but is not required.

**RQ2 · Mechanism.** How is the routing implemented where it exists? Do image-token states encode their own location, or only their content? Which layers and heads of Qwen's language model move layout to the answer position, and does MiniVLM's attention look at the right tokens but merge them without position, or not look at them at all?

- *H2:* In models that route layout, image-token states carry their own location (supplied by 2-D positions or learned), and a small set of mid-layer attention heads transfers it; in MiniVLM the token states carry content but little location.

**RQ3 · Limits.** Does full training close the gap for harder spatial relations as well, or only for the simplest case? Left/right of a single large person is the easiest spatial question; above/below, relations between two objects ("is the dog left of the person"), relative size and orientation may still show a gap in Qwen2.5-VL and similar models.

- *H3:* The gap reappears, partially, for relations that require binding two objects to their locations.

**RQ4 · Measurement.** Can the stage-probe protocol (encoder → projector → image tokens → answer position → calibrated answer) serve as a general diagnostic that separates missing information, failed routing and answer bias, and does it predict benchmark failures such as MMVP's?

---

## 6 · What would have to be shown, in order

These are proposals for after Stage 1; none of them has been run.

| # | Experiment | Answers | Cost |
|---|---|---|---|
| 1 | **Harder spatial tasks** on COCO boxes (above/below, two-object relations, larger-of-two) with the existing pipeline, on MiniVLM and Qwen2.5-VL-7B | RQ3; decides whether a gap remains in fully trained models | Task definitions only |
| 2 | **Self-location probe:** predict each image token's own grid cell from its hidden state, per layer, in both models | RQ2 | Small; reuses the forward passes |
| 3 | MiniVLM variants: **2-D position embeddings**, **spatial questions in training** (on categories other than the test category), both, and **LoRA**; re-run the probes | RQ1 | 4–5 training runs of ~80 min on an L4 |
| 4 | **Attention analysis** from the answer position to the image tokens, in both models | RQ2 | Small |
| 5 | Re-score MMVP and A-OKVQA with **AUC / calibrated readouts**; check whether MMVP failures concentrate on spatial pairs | RQ4; revisits E1.2 | Analysis only |

Experiment 1 decides the scope: if Qwen still shows a gap for harder relations, the problem holds for real models as well as for MiniVLM; if not, the thesis is about how routing is created (RQ1–RQ2), with MiniVLM as the controlled failure case.

---

## 7 · Limits of the current evidence

- **Two models, one contrast.** MiniVLM and Qwen2.5-VL-7B differ in many ways at once (§4); the contrast shows routing *can* exist, not what produces it.
- **One task, one category, one seed.** Position was tested only as left/right of a single person (160 test images, intervals ±7–8 points). Qwen's 100% may also reflect exposure to COCO during training.
- **Linear probes.** They show what a linear readout can use; a property absent to a probe could be present nonlinearly. The language model's own output head is linear, which is why this is the relevant measure, but it is a choice.
- **Image-token probes are generous.** The 4×4 pooling gives the probe each token's location for free, so high scores there show that content is kept, not that the model represents layout (experiment 2 tests this).
- **Prior work may already cover parts of this.** Spatial-reasoning weaknesses of VLMs are widely reported, and 2-D positional schemes exist; the contribution has to be the localisation of the routing step and an account of what creates it. The literature check in §8 comes before committing.

---

## 8 · Reading list to check against

These are the works this problem has to be positioned against. Check the details of each before citing.

- Tong et al., 2024, *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs* (MMVP; CLIP-blind pairs).
- Tong et al., 2024, *Cambrian-1* (vision-centric evaluation and encoder comparisons for MLLMs).
- Kamath, Hessel and Chang, 2023, *What's "up" with vision-language models?* (spatial prepositions in VLMs).
- Rahmanzadehgervi et al., 2024, *Vision language models are blind* (simple geometric and spatial tests).
- Wang et al., 2024, *Qwen2-VL* (M-RoPE: 2-D positions for image tokens).
- Bai et al., 2025, *Qwen2.5-VL Technical Report* (the reference model's training data and positional design).
- Zhao et al., 2021, *Calibrate Before Use* (answer-prior calibration of language models).
- Alain and Bengio, 2016, *Understanding intermediate layers using linear classifier probes*.

---

## 9 · Fallback framings

- **If the routing ingredient turns out to be already explained** (for example, 2-D positions alone suffice and this is documented), the **diagnostic framing** remains: the stage-probe protocol (RQ4) as a method to localise where VLMs lose visual information, with layout routing and answer bias as its case studies.
- **If harder relations show no gap in fully trained models either**, the thesis narrows to the **training-signal framing**: what minimal data and positional design let a frozen-LLM VLM route layout (RQ1), and when during training the routing appears.
