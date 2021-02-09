import multiprocessing
import moviepy.editor as mpy
import numpy as np
import ffmpeg
import cv2
import subprocess as sp
from .bg import remove_many
from more_itertools import take, chunked


def worker(return_dict, batch_number, frame_batch, gpu_batchsize, cpu_batchsize):
    """worker function for processing the batch of frames"""

    # here we send batches that our GPU can handle, let's say 5 at a time

    lst = [None] * cpu_batchsize
    frame_number = 0

    # there are cpu_batchsize items and gpu_batchsize
    for frame_minibatch in chunked(frame_batch, gpu_batchsize):
        
        for frame in remove_many(frame_minibatch, model_name="u2net_human_seg"):
            lst[frame_number] = frame
            frame_number = frame_number + 1
       
    return_dict[batch_number] = lst

def get_input_frames(filepath):
    
    clip = mpy.VideoFileClip(filepath)
    clip_resized = clip.resize(height=320)
    img_number = 0

    for frame in clip_resized.iter_frames(dtype="float"):
        yield frame

command = None
proc = None

def get_output_frames(filepath,
        worker_nodes, 
        cpu_batchsize, 
        gpu_batchsize):

    manager = multiprocessing.Manager()
    
    batch_number = 0
    return_dict = manager.dict()
    jobs = []

    wn = 0

    for mini_batch in chunked(get_input_frames(filepath), cpu_batchsize):

        p = multiprocessing.Process(target=worker, args=(
            return_dict, batch_number, 
            mini_batch, gpu_batchsize, 
            cpu_batchsize))

        jobs.append(p)

        p.start()
        print("job started")

        wn = wn + 1
        batch_number = batch_number + 1

        if wn >= worker_nodes:
            # now sync and wait for jobs to finish

            print("blocking on workers stopping")
            for proc in jobs:
                proc.join()

            wn = 0
            jobs = []
            
            for bn in range(0,worker_nodes):
                for mask in return_dict[bn]:

                    # we use a fixed size data structure, gap at end
                    if mask is not None:
                        yield mask

            return_dict = manager.dict()
            batch_number = 0


def parallel_greenscreen(filepath : str, worker_nodes = 3, cpu_batchsize = 2500, gpu_batchsize = 5):

    command = None
    proc = None

    for image in get_output_frames(
        filepath,
        worker_nodes, 
        cpu_batchsize, 
        gpu_batchsize):

        if command is None: 
            command = ['FFMPEG',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', F"{image.shape[1]},320",
                '-pix_fmt', 'bgr24',
                '-r', "30", # for now I am hardcoding it, I can always resize the clip in premiere anyway 
                '-i', '-',  
                '-an',
                '-vcodec', 'mpeg4',   
                '-b:v', '2000k',    
                filepath.replace(".mp4", ".matte.mp4")  ]
            proc = sp.Popen(command, stdin=sp.PIPE)

        proc.stdin.write(image.tostring())

    proc.stdin.close()
    proc.wait()   