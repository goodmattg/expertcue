import argparse
import numpy as np
import os
import sys
import traceback

from utils.filesystem import file_exists

# Load to Openpose bundles (collected numpy matrices)

# Use the same squeeze as in predict.py

# Encode input video OpenPose bundles (preprocessed for LCM/Transmomo) into character-agnostic encoder

# Run DTW on the encoder over the whole video
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("vid1-bundle", type=file_exists)
    parser.add_argument("vid2-bundle", type=file_exists)

    parser.add_argument("motion-enc-path", type=file_exists)

    parser.add_argument("--source-height", dest="source_height", required=True, type=int, help="source height")
    parser.add_argument("--source-width", dest="source_width", required=True, type=int, help="source width")

    parser.add_argument("--output-fname", dest="output_fname", type=str, default=None, help="output video filename")

    # fmt: on
    args = parser.parse_args()
    return args


def video_dtw(args):

    try:
        # To do things...
        pass

    except:
        print("Unable to render keypoint motion as video!")
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


if __name__ == "__main__":

    args = parse_args()
    video_dtw(args)
