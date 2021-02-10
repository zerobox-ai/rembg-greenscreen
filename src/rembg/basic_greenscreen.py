import moviepy.editor as mpy
import cv2
import subprocess as sp
from .bg import remove_many
from more_itertools import chunked


def get_input_frames(path):
    clip = mpy.VideoFileClip(path)
    clip_resized = clip.resize(height=320)
    img_number = 0

    for frame in clip_resized.iter_frames(dtype="uint8"):
        yield frame

def get_output_frames(gpubatchsize, path):
    for gpu_batch in chunked(get_input_frames(path), gpubatchsize):
        for mask in remove_many(gpu_batch,
                    model_name="u2net_human_seg",
                    compression = False,
                    use_nnserver = False):
            yield mask

def basic_greenscreen(path, gpubatchsize):

    command = None
    proc = None

    for image in get_output_frames(gpubatchsize, path):

        if command is None: 
            command = ['FFMPEG',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', F"{image.shape[1]}x320",
                '-pix_fmt', 'gray',
                '-r', "30", # for now I am hardcoding it, I can always resize the clip in premiere anyway 
                '-i', '-',  
                '-an',
                '-vcodec', 'mpeg4',   
                '-b:v', '2000k',    
                path.replace(".mp4", ".matte.mp4")  ]
            proc = sp.Popen(command, stdin=sp.PIPE)

        proc.stdin.write(image.tostring())

    proc.stdin.close()
    proc.wait()
