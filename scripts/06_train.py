"""Main run with the `model` and `train` sections of config.yaml.

Safe to re-run after an interruption: it resumes from the last checkpoint.
If it runs out of memory, pass `--batch_size 4 --accum 8` (same effective
batch) or `--grad_checkpointing`. Extra arguments are passed to minivlm.train.
"""

from _common import DATA, RUN, module

module("minivlm.train", "--data_dir", DATA, "--out_dir", RUN, "--resume")
