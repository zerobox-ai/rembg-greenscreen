import errno
from io import BytesIO
import io
import json
from multiprocessing import Lock
import os
import sys
import urllib.request
import time
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hsh.library.hash import Hasher
from PIL import Image
from skimage import transform
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import normalize
from . import data_loader, u2net
from skimage.transform import rescale, resize, downscale_local_mean
from timeit import timeit
from funcy import debug
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process

def download_file_from_google_drive(id, fname, destination):
    head, tail = os.path.split(destination)
    os.makedirs(head, exist_ok=True)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    total = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc=f"Downloading {tail} to {head}",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def load_model(model_name: str = "u2net"):
    hasher = Hasher()

    if model_name == "u2netp":
        net = u2net.U2NETP(3, 1)
        path = os.environ.get(
            "U2NETP_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "e4f636406ca4e2af789941e7f139ee2e"
        ):
            download_file_from_google_drive(
                "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
                "u2netp.pth",
                path,
            )

    elif model_name == "u2net_human_seg":
        net = u2net.U2NET(3, 1)
        path = os.environ.get(
            "U2NET_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
        ):
            download_file_from_google_drive(
                "1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
                "u2net_human.pth",
                path,
            )

    elif model_name == "u2net":
        net = u2net.U2NET(3, 1)
        path = os.environ.get(
            "U2NET_PATH",
            os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
        )
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
        ):
            download_file_from_google_drive(
                "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                "u2net.pth",
                path,
            )
    else:
        print("Choose between u2net or u2netp", file=sys.stderr)

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(
                torch.load(
                    path,
                    map_location="cpu",
                )
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_name + ".pth"
        )

    net.eval()

    return net


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn

from functools import wraps
from time import time


def predict(net, items, use_nnserver=False):

    resized = [ transform.resize(image,(320,320)) for image in items ] # expensive
    # note that transform.resize will return values on [0,1] )(float64)
    # so it silently converts them from 255 uint8
    np_arrays = [ np.array(image) for image in resized]
    master_images = np.array(np_arrays).astype(np.float32)

    # move color chanel to second
    master_images = np.moveaxis(master_images, 3, 1)

    if not use_nnserver:
        predict_np = nn_forwardpass(master_images, net)
    else:
        # running in an MPI context, we call a shared NN server
        predict_np = nn_forwardpass_http(master_images)

    imgs = [ Image.fromarray( (predict_np[i, 0, :, :] * 255).astype(np.uint8), mode="L") for i in range(predict_np.shape[0]) ]

    del predict_np

    return imgs

def get_sharedmemory_key():
    return requests.get("http://127.0.0.1:5000/key/").content.decode("utf-8")


lock = Lock()

# takes a stream of compressed (b,3,320,320)
def nn_forwardpass_http(master_images):

    key = get_sharedmemory_key()

    existing_shm = shared_memory.SharedMemory(name=key)
    # we have 10 slots in our numpy 

    np_array = np.ndarray((master_images.shape[0],3,320,320), dtype=np.float32, buffer=existing_shm.buf)
    result = np.ndarray((master_images.shape[0],3,320,320))

    lock.acquire()
    # do everything in here exclusively over all processes
    np_array[:] = master_images
    requests.post("http://127.0.0.1:5000")
    result[:] = np_array
    lock.release()

    return result

def nn_forwardpass(master_images, net):

    inputs_test = None

    with torch.no_grad():

        inputs_test = torch.FloatTensor(
            master_images 
        ).cuda().float()

        d1, _, _, _, _, _, _ = net(inputs_test)

        # d1 == out torch.Size([batch, 1, 320, 320])
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict_np = predict.cpu().detach().numpy()

        del d1, pred, predict, inputs_test

        return predict_np


        