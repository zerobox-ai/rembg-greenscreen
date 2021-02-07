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
import moviepy.editor as mpy
import numpy as np

def batch(iterable, batch_size):
    while batch := list(islice(iterable, batch_size)):
        yield batch

gpu_batch_size = 10

def get_frames():
    clip = mpy.VideoFileClip("C:\\Users\\tim\\Videos\\test\\pedro_domingos-1612544536600-CFR.mp4")
    clip_resized = clip.resize(width=320, height=320)
    img_number = 0
    for frame in tqdm(clip_resized.iter_frames(dtype="uint8")):
        # do something with the frame (a HxWx3 numpy array)

        img_number = img_number + 1

        frame = np.swapaxes(frame, 0,1)

        frame=frame[ 0:320, :,: ]

        #resize it
        #frame = transform.resize( frame, (320, 320) )

        #Image.fromarray(frame, mode="RGB").save( F"C:\\Users\\tim\\Videos\\test\\__{img_number}.jpg" )

        yield frame

w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)

img_number = 0

for gpu_batch in batch(get_frames(), gpu_batch_size):
    for stream in remove_many(gpu_batch):
        fn = F"C:\\Users\\tim\\Videos\\test\\{img_number}.out.png"  
        img_number = img_number + 1
        with open(fn, "wb") as output:
            w(
                output,
                stream[0]
            )
           





 