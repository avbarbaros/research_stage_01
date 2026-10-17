"""Visual-token arithmetic for Qwen2.5-VL.

Qwen2.5-VL cuts an image into 14-px patches and merges 2x2 patches into one
token, so one visual token covers 28x28 pixels. Before that, qwen_vl_utils
resizes the image (keeping its aspect ratio, sides rounded to multiples of 28)
so that its area lies between min_pixels and max_pixels. Setting both to
budget * 28 * 28 fixes the token count at about `budget` for every image,
whatever its original size.
"""

import math

FACTOR = 28  # pixels per visual token side: 14-px patches, 2x2 merge
DEFAULT_MIN_PIXELS = 4 * FACTOR * FACTOR          # qwen_vl_utils defaults
DEFAULT_MAX_PIXELS = 16384 * FACTOR * FACTOR


def smart_resize(height: int, width: int, min_pixels: int, max_pixels: int,
                 factor: int = FACTOR) -> tuple[int, int]:
    """Same rule as qwen_vl_utils.smart_resize."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def visual_tokens(height: int, width: int, min_pixels: int, max_pixels: int) -> int:
    h, w = smart_resize(height, width, min_pixels, max_pixels)
    return (h // FACTOR) * (w // FACTOR)


def budget_pixels(budget: int) -> int:
    return budget * FACTOR * FACTOR


def native_tokens(height: int, width: int) -> int:
    """Tokens under qwen_vl_utils' default limits: roughly the image's own resolution."""
    return visual_tokens(height, width, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
