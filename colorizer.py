# --------------------------------------------------------------------------- #
#                          ДОБАВЛЯЕМ ИМПОРТЫ
# --------------------------------------------------------------------------- #
from typing import Any, Dict, Tuple, Union, List, Optional
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

from runpod.serverless.utils.rp_download import file as rp_file


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """'#FFAABB' -> (255,170,187)"""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def extract_palette_from_image(img: Image.Image,
                               n_colors: int = 5
                               ) -> List[Tuple[int, int, int]]:
    """KMeans по пикселям, возвращает n_colors самых частых RGB"""
    arr = np.array(img.convert("RGB")).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init='auto').fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


def get_palette(payload) -> List[Tuple[int, int, int]]:
    """Возвращает список RGB-цветов палитры"""
    if 'palette_hex' in payload:
        return [hex_to_rgb(h) for h in payload['palette_hex']]
    if 'palette_image_url' in payload:
        img = url_to_pil(payload['palette_image_url'])
        return extract_palette_from_image(img, n_colors=5)
    raise ValueError(
        "Palette not provided: add 'palette_hex' or 'palette_image_url'."
    )


def fill_mask_with_palette(base_img: Image.Image,
                           mask_img: Image.Image,
                           palette: List[Tuple[int, int, int]]) -> Image.Image:
    """
    Заливает области mask_img цветами из palette (циклически).
    mask_img: 0/255 (или 0/1) – белое там, где мебель.
    """
    base_arr = np.array(base_img.convert("RGB"))
    mask_arr = np.array(mask_img.convert("1"))  # True там, где мебель

    coords = np.argwhere(mask_arr)  # [(y,x), ...]
    if coords.size == 0:
        return base_img  # нечего заливать

    for i, (y, x) in enumerate(coords):
        color = palette[i % len(palette)]
        base_arr[y, x] = color

    return Image.fromarray(base_arr)
