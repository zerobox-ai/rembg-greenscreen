import multiprocessing
import moviepy.editor as mpy
import subprocess as sp
from .bg import remove_many
from more_itertools import chunked
import time 
import math
import ffmpeg
import re

def worker(worker_nodes,
            worker_index,
            result_dict, 
            model_name, 
            gpu_batchsize,
            total_frames,
            frames_dict):

    i = 0

    frame_indexes = chunked(range(total_frames),gpu_batchsize)

    print(F"WORKER {worker_index} ONLINE")

    # skip ahead depending on the worker index
    for wi in range(worker_index):
        next(frame_indexes)

    while(fi := next(frame_indexes, False)):

        fi = list(fi)

        # are we processing frames faster than the frame ripper is saving them?
        while fi[-1] not in frames_dict:
            time.sleep(0.1)

        frames = [ frames_dict[index] for index in fi ]

        result_dict[(worker_nodes*i)+(worker_index+1)] = list(remove_many(frames, model_name))

        # clean up the frame buffer
        for fdex in fi:
            del frames_dict[fdex]

        i = i + 1

        # skip ahead
        for wi in range(worker_nodes-1):
           next(frame_indexes, "-1")
     

def process_frame_queue(frame_table, 
    file_path, 
    total_frames, 
    worker_nodes, 
    frame_rate):

    command = None
    proc = None
    hash_index = 0
    frame_counter = 0

    for i in range( math.ceil(total_frames/worker_nodes) ):

        for wx in range(worker_nodes):

            hash_index = i * worker_nodes + 1 + wx

            while hash_index not in frame_table:
                time.sleep(0.01)

            frames = frame_table[hash_index]
            # dont block access to it anymore
            del frame_table[hash_index]

            for frame in frames:
                if command is None: 
                    command = ['FFMPEG',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec','rawvideo',
                        '-s', F"{frame.shape[1]}x320",
                        '-pix_fmt', 'gray',
                        '-r', F"{frame_rate}",
                        '-i', '-',  
                        '-an',
                        '-vcodec', 'mpeg4',   
                        '-b:v', '2000k',    
                        re.sub("\.(mp4|mov|avi)", ".matte.\\1", file_path, flags=re.I)  ]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tostring())
                frame_counter = frame_counter + 1

                if frame_counter >= total_frames: 
                    proc.stdin.close()
                    proc.wait() 
                    print(F"FINISHED ALL FRAMES ({total_frames})!")
                    return

    proc.stdin.close()
    proc.wait() 

def get_input_frames(file_path):
    
    print(F"WORKER FRAMERIPPER ONLINE")

    clip = mpy.VideoFileClip(file_path)
    clip_resized = clip.resize(height=320)

    for frame in clip_resized.iter_frames(dtype="uint8"):
            yield frame

def capture_frames(file_path, frames_dict):
    
    for f in enumerate(get_input_frames(file_path)):
        frames_dict[f[0]] = f[1]

def parallel_greenscreen(file_path, 
    worker_nodes, 
    gpu_batchsize, 
    model_name,
    frame_limit):

    manager = multiprocessing.Manager()

    results_dict = manager.dict()
    frames_dict = manager.dict()

    info = ffmpeg.probe(file_path)
    total_frames = int(info["streams"][0]["nb_frames"])

    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    frame_rate = math.ceil(eval(info["streams"][0]["r_frame_rate"]))

    print(F"FRAME RATE: {frame_rate} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(
        target=capture_frames, args=( file_path, frames_dict )
        ).start()
   
    for wn in range(worker_nodes):
        # note I am deliberatley not using pool
        # we can't trust it to run all the threads concurrently (or at all)
        multiprocessing.Process(target=worker, args=( worker_nodes, wn,
            results_dict, 
            model_name, 
            gpu_batchsize, total_frames, frames_dict)).start()

    process_frame_queue(results_dict, 
        file_path, 
        total_frames, 
        worker_nodes, 
        frame_rate)

    
    