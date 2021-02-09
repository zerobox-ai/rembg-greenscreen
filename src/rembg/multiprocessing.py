import multiprocessing
import moviepy.editor as mpy
import numpy as np
import ffmpeg
import cv2
import subprocess as sp
from .bg import remove_many
from more_itertools import take, chunked
import math

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

def process_batch(batch, no_batches):
    for bn in range(0,no_batches):
        if bn in batch:
            for mask in batch[bn]:

                # we use a fixed size data structure, gap at end
                if mask is not None:
                    yield mask

def get_output_frames(filepath,
        worker_nodes, 
        cpu_batchsize, 
        gpu_batchsize):

    manager = multiprocessing.Manager()
    
   # no_batches = math.ceil(cpu_batchsize/worker_nodes)
    previous_batch = None

    for worker_batch in chunked(get_input_frames(filepath), worker_nodes * cpu_batchsize):

        return_dict = manager.dict()
        jobs = []

        for mini_batch in enumerate(chunked(worker_batch, cpu_batchsize)):

            p = multiprocessing.Process(target=worker, args=(
                return_dict, mini_batch[0], 
                mini_batch[1], gpu_batchsize, 
                cpu_batchsize))

            jobs.append(p)
            p.start()

        # ON PREV BATCH IF PREV BATCH
        # this means we can be busy yielding to FFMPEG
        # while our workers are busy processing new frames
        if previous_batch:
            yield process_batch(previous_batch, worker_nodes)
        
        for proc in jobs:
            proc.join()
            
        previous_batch = return_dict

    # we will have one left over
    yield process_batch(previous_batch, worker_nodes)

def parallel_greenscreen(filepath : str, worker_nodes = 3, cpu_batchsize = 2500, gpu_batchsize = 5):

    command = None
    proc = None

    for gen in get_output_frames(
        filepath,
        worker_nodes, 
        cpu_batchsize, 
        gpu_batchsize):

        for frame in gen:
            if command is None: 
                command = ['FFMPEG',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-s', F"{frame.shape[1]},320",
                    '-pix_fmt', 'gray',
                    '-r', "30", # for now I am hardcoding it, I can always resize the clip in premiere anyway 
                    '-i', '-',  
                    '-an',
                    '-vcodec', 'mpeg4',   
                    '-b:v', '2000k',    
                    filepath.replace(".mp4", ".matte.mp4")  ]
                proc = sp.Popen(command, stdin=sp.PIPE)

            proc.stdin.write(frame.tostring())

    proc.stdin.close()
    proc.wait()  