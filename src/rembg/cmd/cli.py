import argparse
import os

import torch

from ..basic_greenscreen import basic_greenscreen
from ..multiprocessing import parallel_greenscreen


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        default="u2net_human_seg",
        type=str,
        choices=("u2net", "u2netp", "u2net_human_seg"),
        help="The model name.",
        )

    ap.add_argument(
        "-p",
        "--path",
        nargs="+",
        help="Path of a file or a folder of files.",
        )

    ap.add_argument(
        "-g",
        "--greenscreen",
        type=str,
        help="Path of a video.",
        )

    ap.add_argument(
        "-pg",
        "--parallelgreenscreen",
        type=str,
        help="Path of a video for parallel green screen.",
        )

    ap.add_argument(
        "-wn",
        "--workernodes",
        default=1,
        type=int,
        help="Number of parallel workers"
        )

    ap.add_argument(
        "-gb",
        "--gpubatchsize",
        default=4,
        type=int,
        help="GPU batchsize"
        )

    ap.add_argument(
        "-fl",
        "--framelimit",
        default=-1,
        type=int,
        help="Limit the number of frames to process for quick testing.",
        )

    args = ap.parse_args()
    if args.parallelgreenscreen:

        parallel_greenscreen(os.path.abspath(args.parallelgreenscreen),
                             worker_nodes=args.workernodes,
                             gpu_batchsize=args.gpubatchsize,
                             model_name=args.model,
                             frame_limit=args.framelimit)

    elif args.greenscreen:
        basic_greenscreen(
            os.path.abspath(args.greenscreen),
            args.gpubatchsize,
            args.model,
            frame_limit=args.framelimit)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
