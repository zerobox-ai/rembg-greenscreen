import time
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
def remove_many(image_data: typing.List[np.array], net):
    image_data = np.stack(image_data)
    original_shape = image_data.shape[1:3]
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    image_data = torch.transpose(image_data, 1, 3)
    image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
    image_data = (image_data / 255 - 0.485) / 0.229
    out = net(image_data)[:, 0:1]
    ma = torch.max(out)
    mi = torch.min(out)
    dn = (out - mi) / (ma - mi) * 255
    dn = torch.nn.functional.interpolate(dn, original_shape, mode='bilinear')
    dn = dn[:, 0]
    dn = dn.to(dtype=torch.uint8, device=torch.device('cpu')).detach().numpy()
    return dn
