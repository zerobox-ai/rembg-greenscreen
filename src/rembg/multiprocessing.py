import math
import multiprocessing
import os
import re
import subprocess as sp
import time

import ffmpeg
import moviepy.editor as mpy
import requests
import torch
import torch.nn.functional
from hsh.library.hash import Hasher
from tqdm import tqdm

from .bg import DEVICE, remove_many
from .u2net import u2net


def worker(worker_nodes,
           worker_index,
           result_dict,
           model_name,
           gpu_batchsize,
           total_frames,
           frames_dict):
    print(F"WORKER {worker_index} ONLINE")

    output_index = worker_index + 1
    worker_nodesm1 = worker_nodes - 1
    base_index = worker_index * gpu_batchsize
    hasher = Hasher()

    model, hash_val, drive_target, env_var = {
        'u2netp':          (u2net.U2NETP,
                            'e4f636406ca4e2af789941e7f139ee2e',
                            '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                            'U2NET_PATH'),
        'u2net':           (u2net.U2NET,
                            '09fb4e49b7f785c9f855baf94916840a',
                            '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                            'U2NET_PATH'),
        'u2net_human_seg': (u2net.U2NET,
                            '347c3d51b01528e5c6c071e3cff1cb55',
                            '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                            'U2NET_PATH')
        }[model_name]
    path = os.environ.get(env_var, os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")))
    net = model(3, 1)
    if not os.path.exists(path) or hasher.md5(path) != hash_val:
        head, tail = os.path.split(path)
        os.makedirs(head, exist_ok=True)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        response = session.get(URL, params={"id": drive_target}, stream=True)

        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            params = {"id": drive_target, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        total = int(response.headers.get("content-length", 0))

        with open(path, "wb") as file, tqdm(
            desc=f"Downloading {tail} to {head}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
    net.to(DEVICE)
    net.eval()
    for fi in (list(range(base_index + i * worker_nodes,
                          min(base_index + i * worker_nodes + gpu_batchsize, total_frames)))
               for i in range(0, math.ceil(total_frames / worker_nodes), gpu_batchsize)):
        print(fi)
        if not fi:
            break

        # are we processing frames faster than the frame ripper is saving them?
        last = fi[-1]
        while last not in frames_dict:
            time.sleep(0.1)

        result_dict[output_index] = remove_many([frames_dict[index] for index in fi], net)

        # clean up the frame buffer
        for fdex in fi:
            del frames_dict[fdex]
        output_index += worker_nodes

def capture_frames(file_path, frames_dict):
    print(F"WORKER FRAMERIPPER ONLINE")

    for idx, frame in enumerate(mpy.VideoFileClip(file_path).resize(height=320).iter_frames(dtype="uint8")):
        frames_dict[idx] = frame


def parallel_greenscreen(file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1):
    manager = multiprocessing.Manager()

    results_dict = manager.dict()
    frames_dict = manager.dict()

    info = ffmpeg.probe(file_path)
    total_frames = int(info["streams"][0]["nb_frames"])

    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    frame_rate = math.ceil(eval(info["streams"][0]["r_frame_rate"]))

    print(F"FRAME RATE: {frame_rate} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(target=capture_frames, args=(file_path, frames_dict))
    p.start()

    # note I am deliberatley not using pool
    # we can't trust it to run all the threads concurrently (or at all)
    workers = [multiprocessing.Process(target=worker,
                                       args=(worker_nodes, wn, results_dict, model_name, gpu_batchsize, total_frames,
                                             frames_dict))
               for wn in range(worker_nodes)]
    for w in workers:
        w.start()

    command = None
    proc = None
    frame_counter = 0

    for i in range(math.ceil(total_frames / worker_nodes)):
        for wx in range(worker_nodes):

            hash_index = i * worker_nodes + 1 + wx

            while hash_index not in results_dict:
                time.sleep(0.1)

            frames = results_dict[hash_index]
            # dont block access to it anymore
            del results_dict[hash_index]

            for frame in frames:
                if command is None:
                    command = ['FFMPEG',
                               '-y',
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', F"{frame.shape[1]}x320",
                               '-pix_fmt', 'gray',
                               '-r', F"{frame_rate}",
                               '-i', '-',
                               '-an',
                               '-vcodec', 'mpeg4',
                               '-b:v', '2000k',
                               re.sub("\.(mp4|mov|avi)", ".matte.\\1", file_path, flags=re.I)]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tostring())
                frame_counter = frame_counter + 1

                if frame_counter >= total_frames:
                    proc.stdin.close()
                    proc.wait()
                    print(F"FINISHED ALL FRAMES ({total_frames})!")
                    return

    p.join()
    for w in workers:
        w.join()

    proc.stdin.close()
    proc.wait()
