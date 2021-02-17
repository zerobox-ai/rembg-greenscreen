import re
import subprocess as sp

from more_itertools import chunked

from .bg import get_model, iter_frames, remove_many


def basic_greenscreen(path, gpubatchsize, model_name, dtype, frame_limit=-1):
    command = None
    proc = None
    net = get_model(model_name, dtype)
    for gpu_batch in chunked(iter_frames(path), gpubatchsize):
        if 0 < frame_limit < gpubatchsize:
            break
        frame_limit -= gpu_batch
        for image in remove_many(gpu_batch, net, dtype):
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
                           re.sub(r"\.(mp4|mov|avi)", r".matte.\1", "tim.mov", flags=re.I)]
                proc = sp.Popen(command, stdin=sp.PIPE)
            proc.stdin.write(image.tostring())
    if isinstance(proc, sp.Popen):
        proc.stdin.close()
        proc.wait()
