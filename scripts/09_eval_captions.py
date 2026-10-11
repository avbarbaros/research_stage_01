"""Captions for COCO val2017 images, scored with CIDEr (`eval` section of config.yaml).

CIDEr here uses a simple tokeniser, so it is a baseline for E1.2 comparisons,
not a number comparable to published ones. Extra arguments are passed to
minivlm.eval."""

from _common import DATA, RUN, module

module("minivlm.eval", "caption", "--data_dir", DATA, "--out_dir", RUN)
