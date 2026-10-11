"""Main-run data, sized by the `data` section of config.yaml: detail
descriptions plus conversations, and only the COCO images they need. Extra
arguments are passed to minivlm.data."""

from _common import DATA, module

module("minivlm.data", "--data_dir", DATA)
