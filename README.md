# research_stage_01
Multimodal reasoning models Stage-1

## E1.1 Minimal VLM

A hand-written VLM: frozen vision encoder → trainable projector → frozen LLM. Only the projector is trained, on a 50k subset of LLaVA-Instruct (detail descriptions + conversations) with COCO images.

| Part | Default |
|---|---|
| Vision encoder | `openai/clip-vit-large-patch14`, 224 px, penultimate layer, CLS dropped → 256 tokens × 1024 |
| Projector | `Linear(1024→1536) → GELU → Linear(1536→1536)` |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` |

### Run from a terminal (Colab L4)

```bash
git clone https://github.com/avbarbaros/research_stage_01.git && cd research_stage_01
python scripts/01_setup.py           # GPU check, pip install, show data/runs paths
python scripts/02_run_tests.py       # CPU unit tests
python scripts/03_smoke_test.py      # 500 samples, 0.5B LLM, 30 steps
python scripts/04_trace_shapes.py    # tensor shapes at the vision-language interface
python scripts/05_prepare_data.py    # 50k subset + COCO images
python scripts/06_train.py           # main run; re-run to resume
python scripts/07_plot_loss.py       # -> <run>/loss.png
python scripts/08_eval_grounding.py  # correct vs shuffled vs blank image
python scripts/09_eval_captions.py   # 200 COCO val captions + CIDEr
python scripts/10_show_results.py    # results + <run>/captions_grid.png
```

### Settings

All paths and hyper-parameters are in [config.yaml](config.yaml). Precedence is command-line flag > environment variable (paths only) > `config.yaml`.

- One-off override: scripts 03, 05, 06, 08 and 09 pass extra arguments to the underlying module, e.g. `python scripts/06_train.py --batch_size 4 --accum 8`.
- Another settings file: add `--config other.yaml` to any script or module.
- Each run saves the model and optimisation settings it used to `<run>/config.json`; evaluation reads the model from there, not from `config.yaml`.

#### `paths`

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `/content/data` | Datasets and images, on local disk. Env: `DATA_DIR`. |
| `runs_dir` | `/content/drive/MyDrive/MMRM/stage_01/e1_1_minivlm` | Checkpoints, logs, results. Env: `RUNS_DIR`. |
| `runs_dir_fallback` | `runs` | Used when `runs_dir` is on Google Drive and Drive is not mounted. |
| `run_name` | `main_clip_qwen1.5b` | Main run directory inside `runs_dir`. Env: `RUN_NAME`. |

Relative paths are resolved against the repository root. Drive can only be mounted from a notebook cell (`from google.colab import drive; drive.mount('/content/drive')`), not from the terminal. The defaults are Colab paths; elsewhere, set `DATA_DIR` or edit the file.

#### `model`

| Key | Default | Meaning |
|---|---|---|
| `encoder` | `clip` | `clip`, `siglip`, `dinov2`, or `clip-b32` (small, for CPU debugging). Model ids are in [minivlm/config.py](minivlm/config.py). |
| `llm` | `Qwen/Qwen2.5-1.5B-Instruct` | Frozen language model. |
| `feature_layer` | `-2` | Vision encoder hidden state fed to the projector (penultimate). |
| `projector` | `mlp2` | `mlp2` (two-layer MLP) or `linear`. |
| `projector_hidden` | `0` | Hidden width of the MLP; `0` means the LLM hidden size, a small value makes a bottleneck. |
| `grad_checkpointing` | `false` | Slower, less GPU memory. |

#### `data`

| Key | Default | Meaning |
|---|---|---|
| `subset` | `main` | Name of the prepared subset inside `data_dir`. |
| `n_total` | `50000` | Training samples. |
| `n_holdout` | `500` | Held-out detail descriptions, split by image. |
| `detail_fraction` | `0.46` | Share of detail descriptions; the rest are conversations. |
| `max_text_tokens` | `256` | Token budget per sample, image tokens not counted. |
| `seed` | `0` | Seed for sampling the subset. |
| `download_workers` | `32` | Parallel COCO image downloads. |

#### `train`

| Key | Default | Meaning |
|---|---|---|
| `lr` | `1.0e-3` | Peak learning rate (AdamW, cosine decay). |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.03` | Fraction of optimiser steps used for linear warmup. |
| `batch_size` | `8` | Samples per forward pass. 16 does not fit a 22 GB L4 with the 1.5B LLM. |
| `accum` | `4` | Gradient accumulation; effective batch is `batch_size * accum`. |
| `epochs` | `1` | |
| `max_grad_norm` | `1.0` | Gradient clipping. |
| `seed` | `0` | Seed for initialisation and data order. |
| `save_every` | `500` | Checkpoint interval, in optimiser steps. |
| `log_every` | `10` | Logging interval, in optimiser steps. |
| `num_workers` | `4` | DataLoader workers. |

#### `smoke`

Used only by `scripts/03_smoke_test.py`; anything not listed here comes from `model` and `train`.

| Key | Default | Meaning |
|---|---|---|
| `subset` | `smoke` | Subset name for the smoke data. |
| `run_name` | `smoke` | Run directory inside `runs_dir`. |
| `llm` | `Qwen/Qwen2.5-0.5B-Instruct` | Smaller LLM for a quick check. |
| `n_total` / `n_holdout` | `500` / `50` | Subset size. |
| `batch_size` / `accum` | `8` / `2` | |
| `max_steps` | `30` | Optimiser steps before stopping. |
| `log_every` | `5` | |

#### `eval`

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `16` | |
| `num_workers` | `4` | DataLoader workers for the grounding control. |
| `caption_n` | `200` | COCO val2017 images to caption. |
| `max_new_tokens` | `128` | Generation length for the detailed-description prompt. |
| `cider` | `true` | Score short captions with CIDEr (simple tokeniser, not comparable to published numbers). |
| `seed` | `0` | Seed for picking the val images. |

### Underlying modules

```bash
python -m minivlm.data                                  # subset + images
python -m minivlm.train --out_dir runs/main --resume    # train projector
python -m minivlm.eval grounding --out_dir runs/main    # correct vs shuffled vs blank image
python -m minivlm.eval caption   --out_dir runs/main
pytest tests                                            # CPU, tiny random models
```

Defaults come from `config.yaml`; every setting also has a flag (`--help`). Useful training flags: `--encoder {clip,siglip,dinov2}` (E1.2), `--projector linear`, `--projector_hidden 64` (bottleneck), `--batch_size 4 --accum 8` or `--grad_checkpointing` if memory is short. `--encoder clip-b32` is a small encoder for debugging on CPU.

### Layout

- [minivlm/model.py](minivlm/model.py) — `MiniVLM`; the tensor shapes at the vision–language interface are listed at the top of the file.
- [minivlm/data.py](minivlm/data.py) — subset builder, COCO download, chat-format tokenisation and label masking.
- [minivlm/train.py](minivlm/train.py), [minivlm/eval.py](minivlm/eval.py), [minivlm/config.py](minivlm/config.py)
