import argparse
import glob
import io
import os
from distutils.util import strtobool
from PIL import Image
import filetype
from tqdm import tqdm
from .u2net.detect import predict
from .bg import remove, remove_many
from itertools import islice, chain

r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)

full_paths = ["C:\\Users\\tim\\Videos\\test"]
files = set()

batch_size = 20

for path in full_paths:
    files = (set(glob.glob(path + "/*")) - set(glob.glob(path + "/*.out.png")))

def batch(iterable, batch_size):
    while batch := list(islice(iterable, batch_size)):
        yield batch

def get_files():
    for fi in tqdm(files):
        fi_type = filetype.guess(fi)

        if fi_type is None:
            continue
        elif fi_type.mime.find("image") < 0:
            continue

        with open(fi, "rb") as inp:
            yield fi, Image.open(io.BytesIO(r(inp))).convert("RGB")

for batch in batch(get_files(), batch_size):
    for stream in remove_many(batch):
        fn = os.path.splitext(stream[1])[0] + ".out.png"
        with open(fn, "wb") as output:
            w(
                output,
                stream[0]
            )
           





 