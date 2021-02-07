import functools
import io

import numpy as np
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion

from .u2net import detect


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout

def naive_cutout(img, mask):

    pil_image = Image.fromarray(img, mode="RGB")

    empty = Image.new("RGBA", (pil_image.size), 0)
    # this is an argument to mask on the high res version, you get bicubic interpolation
    # on the geometry, I hadn't thought of that, higher cost though N^2
    cutout = Image.composite(pil_image, empty, mask.resize(pil_image.size, Image.LANCZOS))
    return cutout

@functools.lru_cache(maxsize=None)
def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    elif (model_name == "u2net_human_seg"):
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")

def remove(
    data,
    model_name="u2net_human_seg",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
):
    model = get_model(model_name)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    mask = detect.predict(model, np.array(img)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")

    return bio.getbuffer()


def remove_many(
    image_data,
    model_name="u2net_human_seg"
):
    model = get_model(model_name)
    
    master_images = detect.preprocess(image_data)

    # these are also PIL images
    masks = detect.predict(model, master_images )

    for combo in zip( image_data, masks ):
        # this costs us about 2 images/s 
        cutout = naive_cutout(combo[0], combo[1])

        bio = io.BytesIO()
        cutout.save(bio, "PNG")
        yield (bio.getbuffer(), combo)

