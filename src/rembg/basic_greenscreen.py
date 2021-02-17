import re
import os
import re
import subprocess as sp

import moviepy.editor as mpy
import requests
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from more_itertools import chunked
from tqdm import tqdm

from .bg import DEVICE, remove_many
from .u2net import u2net


def basic_greenscreen(path, gpubatchsize, model_name, frame_limit=-1):
    command = None
    proc = None
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
    for gpu_batch in chunked(mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8"), gpubatchsize):
        if 0 < frame_limit < gpubatchsize:
            break
        frame_limit -= gpu_batch
        for image in remove_many(gpu_batch, net):
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
