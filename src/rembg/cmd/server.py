import argparse
from io import BytesIO
import io
from urllib.parse import unquote_plus
from urllib.request import urlopen
from flask import Flask, request, send_file
import numpy as np
from waitress import serve
from ..u2net.detect import nn_forwardpass
from ..bg import get_model, remove_many
import datetime

app = Flask(__name__)

net = get_model("u2net_human_seg")

@app.route("/", methods=["GET", "POST"])
def index():
    file_content = ""

    if request.method == "POST":
        if "file" not in request.files:
            return {"error": "missing post form param 'file'"}, 400

        file_content = request.files["file"].read()

    if file_content == "":
        return {"error": "File content is empty"}, 400

    try:

        load_bytes = io.BytesIO(file_content)
        load_bytes.seek(0)
        decompressed_array = np.load(load_bytes, allow_pickle=True)

        masks = nn_forwardpass(decompressed_array, net)

        print( F"{datetime.datetime.now()}: sent {masks.shape[0]} images" )

        stream = io.BytesIO()  
        np.save(stream, masks)
        stream.seek(0) 

        return send_file(stream,
            mimetype="application/octet-stream",
        )
    except Exception as e:
        app.logger.exception(e, exc_info=True)
        return {"error": "oops, something went wrong!"}, 500


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-a",
        "--addr",
        default="0.0.0.0",
        type=str,
        help="The IP address to bind to.",
    )

    ap.add_argument(
        "-p",
        "--port",
        default=5000,
        type=int,
        help="The port to bind to.",
    )

    args = ap.parse_args()

    serve(app, host=args.addr, port=args.port, threads=4)


if __name__ == "__main__":
    main()
