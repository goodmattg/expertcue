import argparse
import numpy as np
import os
import sys
import traceback

from dtw import *
from scipy.spatial.distance import cdist
from utils.filesystem import path_exist

from utils.motion import (
    preprocess_motion2d,
    postprocess_motion2d,
    openpose2motion,
    prebundle_openpose_to_motion,
)

from common import config

# Load to Openpose bundles (collected numpy matrices)

# Use the same squeeze as in predict.py

# Encode input video OpenPose bundles (preprocessed for LCM/Transmomo) into character-agnostic encoder

# Run DTW on the encoder over the whole video
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("vid1", type=path_exist)
    parser.add_argument("vid2", type=path_exist)

    parser.add_argument(
        "-mp",
        "--model_path",
        type=path_exist,
        required=True
    )

    parser.add_argument(
        "-v1s",
        "--v1-shape",
        nargs=2,
        metavar=("height", "width"),
        required=True,
        help="video 1 shape: [H, W]",
    )

    parser.add_argument(
        "-v2s",
        "--v2-shape",
        nargs=2,
        metavar=("height", "width"),
        required=True,
        help="video 2 shape: [H, W]",
    )

    parser.add_argument("--vid-npy", action="store_true", help="load OpenPose keypoints from [15, 2, T] numpy (.npy) matrix")    
    # fmt: on
    args = parser.parse_args()

    config.initialize(args)

    return args, config


def video_dtw(args, config):

    try:

        # resize input
        h1, w1, scale1 = pad_to_height(
            config.img_size[0], args.img1_height, args.img1_width
        )
        h2, w2, scale2 = pad_to_height(
            config.img_size[0], args.img2_height, args.img2_width
        )

        if args.vid_npy:
            # Videos are *.npy matrix files
            input1 = prebundle_openpose_to_motion(args.vid1, scale=scale1)
            input2 = prebundle_openpose_to_motion(args.vid2, scale=scale2)
        else:
            # Videos are directories containing *.json files
            input1 = openpose2motion(args.vid1, scale=scale1)
            input2 = openpose2motion(args.vid2, scale=scale2)

        # load trained model
        net = get_autoencoder(config)
        net.load_state_dict(torch.load(args.model_path))
        net.to(config.device)
        net.eval()

        # mean/std pose
        mean_pose, std_pose = get_meanpose(config)

        x1, x2 = args.vid1_bundle, args.vid2_bundle
        # To do things...
        cost = cdist(x1, x2, metric="euclidean")
        alignment = dtw(x=cost, keep_internals=True)

    except:
        print("Unable to render keypoint motion as video!")
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


if __name__ == "__main__":

    args, config = parse_args()
    video_dtw(args, config)
