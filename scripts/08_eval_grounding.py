"""Grounding control: loss on held-out descriptions (images never seen in
training) with the correct image, another sample's image, and a blank image.
If `correct` is not clearly below the other two, the LLM is ignoring the
visual tokens. Extra arguments are passed to minivlm.eval."""

from _common import DATA, RUN, module

module("minivlm.eval", "grounding", "--data_dir", DATA, "--out_dir", RUN)
