import argparse
import glob
import io
import os
from distutils.util import strtobool
from itertools import islice, chain
from PIL import Image
import filetype
from tqdm import tqdm

from ..bg import remove, remove_many


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        default="u2net",
        type=str,
        choices=("u2net", "u2netp"),
        help="The model name.",
    )

    ap.add_argument(
        "-a",
        "--alpha-matting",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="When true use alpha matting cutout.",
    )

    ap.add_argument(
        "-af",
        "--alpha-matting-foreground-threshold",
        default=240,
        type=int,
        help="The trimap foreground threshold.",
    )

    ap.add_argument(
        "-ab",
        "--alpha-matting-background-threshold",
        default=10,
        type=int,
        help="The trimap background threshold.",
    )

    ap.add_argument(
        "-ae",
        "--alpha-matting-erode-size",
        default=10,
        type=int,
        help="Size of element used for the erosion.",
    )

    ap.add_argument(
        "-az",
        "--alpha-matting-base-size",
        default=1000,
        type=int,
        help="The image base size.",
    )

    ap.add_argument(
        "-p",
        "--path",
        nargs="+",
        help="Path of a file or a folder of files.",
    )

    ap.add_argument(
        "-b",
        "--batchsize",
        default=15,
        nargs="+",
        type=int,
        help="GPU batchsize for the GPU"
    )

    ap.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="-",
        type=argparse.FileType("wb"),
        help="Path to the output png image.",
    )

    ap.add_argument(
        "input",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to the input image.",
    )

    args = ap.parse_args()

    r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
    w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)

    if args.path:
        full_paths = [os.path.abspath(path) for path in args.path]
        
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

        for gpu_batch in batch(get_files(), args.batchsize):
            for stream in remove_many(gpu_batch):
                fn = os.path.splitext(stream[1][0][0])[0] + ".out.png"
                with open(fn, "wb") as output:
                    w(
                        output,
                        stream[0]
                    )

    else:
        w(
            args.output,
            remove(
                r(args.input),
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ),
        )


if __name__ == "__main__":
    main()
