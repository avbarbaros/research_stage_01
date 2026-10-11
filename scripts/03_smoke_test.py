"""Smoke test with the `smoke` section of config.yaml: a few hundred samples,
the 0.5B LLM, a few dozen optimiser steps.

Check three things in the output before the main run: the loss goes down,
`peak` memory leaves headroom on the GPU, and `samples/s` gives an acceptable
time for the full subset. Extra arguments are passed to minivlm.train.
"""

import sys

from _common import CFG, DATA, SMOKE_RUN, module, run

s = CFG["smoke"]
run(sys.executable, "-m", "minivlm.data", "--data_dir", DATA, "--subset", s["subset"],
    "--n_total", s["n_total"], "--n_holdout", s["n_holdout"])
module("minivlm.train", "--data_dir", DATA, "--subset", s["subset"], "--out_dir", SMOKE_RUN,
       "--llm", s["llm"], "--batch_size", s["batch_size"], "--accum", s["accum"],
       "--max_steps", s["max_steps"], "--log_every", s["log_every"])
