import argparse
import glob
import os
from distutils.util import strtobool
import filetype
from tqdm import tqdm
from .u2net.detect import predict
from .bg import remove, remove_many

r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)


full_paths = ["C:\\Users\\tim\\Videos\BonsaiVideo\\f_keith_duggar-1608646284491.mp4"]
files = set()

for path in full_paths:
    full_paths += (set(glob.glob(path + "/*")) - set(glob.glob(path + "/*.out.png")))

files = {}


for fi in full_paths[1:20]:

    fi_type = filetype.guess(fi)

    if fi_type is None:
        continue
    elif fi_type.mime.find("image") < 0:
        continue

    with open(fi, "rb") as inp:
        files[fi] = r(inp)

remove_many( files )


input("Press Enter to continue...")
input("Press Enter to continue...")
input("Press Enter to continue...")