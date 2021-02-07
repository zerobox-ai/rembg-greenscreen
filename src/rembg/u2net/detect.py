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
from sklearn.preprocessing import normalize
from . import data_loader, u2net
from skimage.transform import rescale, resize, downscale_local_mean

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

# I have tried pretty much turning this off, it makes no difference to throughput
def preprocess(items):    

    arrays = [np.array((image[1].resize((320,320)))) for image in items ]

    # color grading code which was in ToTensorLab
    # the NN was probably quite sensitive to the normalisation
    # the performance degrades significantly without this
    master_images = np.array(arrays) / 256
   # master_images[:, :, 0] = (master_images[:, :, 0] - np.mean(master_images[:, :, 0])) / np.std(master_images[:, :, 0])
   # master_images[:, :, 1] = (master_images[:, :, 1] - np.mean(master_images[:, :, 1])) / np.std(master_images[:, :, 1])
    #master_images[:, :, 2] = (master_images[:, :, 2] - np.mean(master_images[:, :, 2])) / np.std(master_images[:, :, 2])

    # change the r,g,b to b,r,g from [0,255] to [0,1]
    #master_images = (master_images - np.min(master_images)) / (np.max(master_images)-np.min(master_images))
    
    #RGB->BGR
    #master_images = master_images[:,:,:,::-1]

    # move color chanel to second
    master_images = np.moveaxis(master_images, 3, 1)

    return master_images

def predict(net, master_images):

    batch = master_images.shape[0]

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
               master_images 
            ).cuda().float()

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # d1 == out torch.Size([batch, 1, 320, 320])
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

       # predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()

        imgs = [ Image.fromarray(predict_np[i, :, :] * 256, mode="L") for i in range(batch) ]

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test

        torch.cuda.empty_cache()

        return imgs