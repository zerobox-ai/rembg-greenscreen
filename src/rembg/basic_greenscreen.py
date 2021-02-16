import re
import subprocess as sp

import moviepy.editor as mpy
from more_itertools import chunked

from .bg import remove_many


def basic_greenscreen(path, gpubatchsize, model_name, frame_limit=-1):
    command = None
    proc = None

    for gpu_batch in chunked(mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8"), gpubatchsize):
        if 0 < frame_limit < gpubatchsize:
            break
        frame_limit -= gpu_batch
        for image in remove_many(gpu_batch, model_name):
            if command is None:
                command = ['FFMPEG',
                           '-y',
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-s', F"{image.shape[1]}x320",
                           '-pix_fmt', 'gray',
                           '-r', "30",  # for now I am hardcoding it, I can always resize the clip in premiere anyway
                           '-i', '-',
                           '-an',
                           '-vcodec', 'mpeg4',
                           '-b:v', '2000k',
                           re.sub("\.(mp4|mov|avi)", ".matte.\\1", "tim.mov", flags=re.I)]
                proc = sp.Popen(command, stdin=sp.PIPE)
            proc.stdin.write(image.tostring())
    if isinstance(proc, sp.Popen):
        proc.stdin.close()
        proc.wait()
