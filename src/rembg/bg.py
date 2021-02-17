import os
import typing

import numpy as np
import requests
import torch
import torch.nn.functional
from hsh.library.hash import Hasher
from tqdm import tqdm

from .u2net import u2net

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def remove_many(image_data: typing.List[np.array], model_name: str):
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
    net.load_state_dict(torch.load(path))
    net.to(DEVICE)
    net.eval()
    image_data = np.stack(image_data)
    original_shape = image_data.shape[1:]
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='linear')
    image_data = (image_data / 255 - 0.485) / 0.229
    image_data = torch.transpose(image_data, 1, 3)
    out = net(image_data)[:, 0, :, :]
    ma = torch.max(out)
    mi = torch.min(out)
    dn = (out - mi) / (ma - mi) * 255
    dn = torch.nn.functional.interpolate(dn, original_shape, mode='linear')
    dn = dn.to(dtype=torch.uint8, device=torch.device('cpu')).detach().numpy()
    return dn
