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

def batch(iterable, batch_size):
    while batch := list(islice(iterable, batch_size)):
        yield batch

width = None

def get_input_frames():
    clip = mpy.VideoFileClip("C:\\Users\\tim\\Videos\\test\\2021-01-30 20-28-16.mp4")
    clip_resized = clip.resize(height=320)
    img_number = 0

    global width

    for frame in clip_resized.iter_frames(dtype="float"):

        if width is None: 
            width=frame.shape[0]

        yield frame

def get_output_frames():
    for gpu_batch in batch(get_input_frames(), gpu_batch_size):
        for mask in remove_many(gpu_batch):
            yield mask

output_file = 'C:\\Users\\tim\\Videos\\test\\output_file_name.mp4'
input_file = 'C:\\Users\\tim\\Videos\\test\\2021-01-30 20-28-16.mp4'

command = ['FFMPEG',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', F"320,568",
        '-pix_fmt', 'bgr24',
        '-r', "24", 
        '-i', '-',  
        '-an',
        '-vcodec', 'mpeg4',   
        '-b:v', '2000k',    
        output_file ]

proc = sp.Popen(command, stdin=sp.PIPE)
video = None


for image in get_output_frames():

    if video is None:
        dimension = '{}x{}'.format(image.shape[0], image.shape[1])

    proc.stdin.write(image.tostring())


proc.stdin.close()
proc.wait()