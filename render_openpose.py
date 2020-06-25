import argparse
import numpy as np
import os
import sys
import traceback

from utils.visualization import motion2video, hex2rgb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("input_fname", default=None, type=str)

    parser.add_argument("--source-height", dest="source_height", required=True, type=int, help="source height")
    parser.add_argument("--source-width", dest="source_width", required=True, type=int, help="source width")

    parser.add_argument("--output-fname", dest="output_fname", type=str, default=None, help="output video filename")

    parser.add_argument("--color", type=str, default="#a50b69#b73b87#db9dc3", help="skeleton render color")

    # fmt: on
    args = parser.parse_args()
    return args


def render_keypoints(args):

    try:
        with open(args.input_fname, "rb") as f:

            motion = np.load(f)

            color = hex2rgb(args.color)

            output_fname = (
                args.output_fname
                if args.output_fname
                else "{}_mx.mp4".format(
                    os.path.splitext(os.path.basename(args.input_fname))[0]
                )
            )

            motion2video(
                motion,
                args.source_height,
                args.source_width,
                output_fname,
                color,
                transparency=False,
            )
    except:
        print("Unable to render keypoint motion as video!")
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


if __name__ == "__main__":

    args = parse_args()
    render_keypoints(args)
