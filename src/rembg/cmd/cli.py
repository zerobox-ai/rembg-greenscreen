import argparse
from itertools import islice
from ..basic_greenscreen import basic_greenscreen
from ..multiprocessing import parallel_greenscreen
from ..bg import remove_many

def batch(iterable, batch_size):
    while batch := list(islice(iterable, batch_size)):
        yield batch

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
        default=4,
        type=int,
        help="GPU batchsize"
    )

    ap.add_argument(
        "-gb",
        "--gpubatchsize",
        default=2,
        type=int,
        help="GPU batchsize"
    )

    ap.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="-",
        type=str,
        help="Path to the output png image.",
    )

    ap.add_argument(
            "-nns",
            "--nnserver",
            default=False,
            type=bool,
            help="Use the NN HTTP server?",
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

    if args.parallelgreenscreen:

        parallel_greenscreen(args.parallelgreenscreen, 
            worker_nodes = args.workernodes, 
            gpu_batchsize = args.gpubatchsize,
            model_name = args.model)

    elif args.greenscreen:
        basic_greenscreen(
            args.greenscreen, 
            args.gpubatchsize,
            args.model)
      

if __name__ == "__main__":
    main()
