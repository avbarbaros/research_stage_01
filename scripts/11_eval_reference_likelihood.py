"""Loss of the COCO reference captions of the same val images used by
09_eval_captions.py, given the short-caption prompt, with the correct image and
with another image. No generation: it measures whether the visual tokens make
the human captions more likely. Extra arguments are passed to minivlm.eval."""

from _common import DATA, RUN, module

module("minivlm.eval", "reference", "--data_dir", DATA, "--out_dir", RUN)
