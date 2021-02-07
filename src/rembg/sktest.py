import argparse
import glob
import io
import os
from distutils.util import strtobool
from PIL import Image
import filetype
from skimage import transform
from tqdm import tqdm
from .u2net.detect import predict
from .bg import remove, remove_many
from itertools import islice, chain
import numpy as np

r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)


image = Image.open(io.BytesIO(r(open("C:\\Users\\tim\\Videos\\test\\50.jpg", "rb")))).convert("RGB")

image = image.resize((300,300), Image.ANTIALIAS)

Image.fromarray( np.array(image), mode="RGB" ).save("C:\\Users\\tim\\Videos\\test\\tim.jpg")



           





 