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
import cv2
import ffmpeg
import cv2
import subprocess as sp

gpu_batch_size = 10


clip = mpy.VideoFileClip("C:\\Users\\tim\\Videos\\test\\2021-01-30 20-28-16.mp4")
clip_resized = clip.resize(height=320)

x=0

for frame in tqdm(clip_resized.iter_frames(dtype="uint8")):
    x=x+1


print(x)

