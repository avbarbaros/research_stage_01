"""Multiple-choice benchmarks scored by option likelihood: MMVP (fine-grained
perception, CLIP-blind image pairs) and A-OKVQA (VQA). Each option is scored
with the real image and with a blank one, so you can see what the image adds
over the language prior. Re-run 12_compare_runs.py afterwards to compare runs.
Extra arguments are passed to minivlm.eval, e.g. `--benchmarks mmvp`."""

from _common import RUN, module

module("minivlm.eval", "options", "--out_dir", RUN)
