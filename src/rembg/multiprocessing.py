import multiprocessing
import moviepy.editor as mpy
import numpy as np
import ffmpeg
import cv2
import subprocess as sp
from .bg import remove_many
from .cmd.cli import batch

def worker(return_dict, batch_number, frame_batch):
    """worker function for processing the batch of frames"""

    # here we send batches that our GPU can handle, let's say 5 at a time

    lst = [None] * 100
    frame_number = 0
    # note that 3 means in this batch, there are another 4 guys like this
    # so 4*3=12 images on gpu and loading model 4 times
    #frame_batch = [F,F,F...,100]
    for frame_minibatch in batch(frame_batch, 2):
        for frame in remove_many(frame_minibatch, model_name="u2netp"):
            lst[frame_number] = frame
            frame_number = frame_number + 1
       
    return_dict[batch_number] = lst

def get_input_frames():
    
    clip = mpy.VideoFileClip(filepath)
    clip_resized = clip.resize(height=320)
    img_number = 0

    for frame in clip_resized.iter_frames(dtype="float"):
        yield frame

command = None
proc = None

def get_output_frames():

    #manager = multiprocessing.Manager()
    
    #0-2000, 2000-4000, 4000-6000, 
    #1,.....,2........,.3..........
    for frame_batch in batch(get_input_frames(), 50):

        batch_number = 1
        #return_dict = manager.dict()
        #jobs = []

        print("big batch")
        input("big batch")

        for mini_batch in batch(frame_batch, 25):
            # we will have 4 batches here

            #0-500, 500-1000,1000-1500, 1500-2000
            #1......2..........3.........4

            #p = multiprocessing.Process(target=worker, args=(return_dict, batch_number, mini_batch))
            #jobs.append(p)
            #p.start()

            batch_number = batch_number + 1

            print(batch_number)

        print("out of loop")
        input("test")

        # now sync and wait for jobs to finish
        #for proc in jobs:
        #    proc.join()

        # now reintegrate
        #for b in range(1,batch_number):
        #    for mask in frame_batch[b]:
        #        yield mask

if __name__ == "__main__":

    filepath = "C:\\Users\\tim\\Videos\\AWS\\bothsofian\\Sofian\\sofian-1608288312764_CFR.mp4"

    for image in get_output_frames():

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