import functools
import numpy as np
from PIL import Image
from .u2net import detect

@functools.lru_cache(maxsize=None)
def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    elif (model_name == "u2net_human_seg"):
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")

def remove_many(
    image_data,
    model_name
):

    orig_height = image_data[0].shape[0]
    orig_width = image_data[0].shape[1]

    model = get_model(model_name)

    # these are also PIL images
    masks = detect.predict(model, image_data )

    for mask in masks:

        # resize back to original aspect (from square)
        mask = mask.resize( (orig_width, orig_height), Image.LANCZOS)
        mask = np.array( mask ).astype(np.uint8)
        
        yield mask