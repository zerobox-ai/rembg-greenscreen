import errno
import os
import sys
import urllib.request

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

from . import data_loader, u2net


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


def predict(net, items):

    images = np.zeros( (19, 320, 320) )

    arrays = [np.array(image) for image in items ]

    master_images = np.array(arrays)[:,0:320, 0:320]
 
    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
                np.expand_dims(master_images, 0)
            ).cuda().float()

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = Image.fromarray(predict_np * 255).convert("RGB")

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img
