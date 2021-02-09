
from io import BytesIO
import io
import pathlib
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from ..u2net.detect import nn_forwardpass_http


image = "C:\\Users\\tim\\Pictures\\Screenshot 2020-11-07 224940.png"



for i in tqdm(range(100)):

    stream = io.BytesIO()    
    np.savez_compressed(stream, np.zeros( (10,3,320,320)) )
    stream.seek(0)

    nn_forwardpass_http( stream )
    print(i)





