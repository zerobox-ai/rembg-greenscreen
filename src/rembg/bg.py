import functools
import io
import numpy as np
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
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
    model_name="u2net_human_seg",
    compression = False,
    use_nnserver = False
):

    orig_height = image_data[0].shape[0]
    orig_width = image_data[0].shape[1]

    model = None

    if not use_nnserver:
        model = get_model(model_name)

    if compression:
        for arr in image_data:
            arr.seek(0)

        image_data = [np.load(compressed_array,allow_pickle=True)['arr_0'] for compressed_array in image_data]

    # these are also PIL images
    masks = detect.predict(model, image_data, use_nnserver )

    for mask in masks:

        mask = mask.resize( (orig_width, orig_height), Image.LANCZOS)
        mask = np.array( mask ).astype(np.uint8)
        
        if compression:
            # compress it in storage (trade CPU which we have plenty of for RAM which is in short supply)
            # Probably usless unless you have running MANY workernodes
            compressed_array = io.BytesIO()    
            np.savez_compressed(compressed_array, mask)
            yield compressed_array
        
        else:
            yield mask

