import argparse
import numpy as np
import os
import sys
import traceback
import torch
import pdb
import dtw
import imageio
import random, string
import torchvision


from scipy.spatial.distance import cdist
from utils.filesystem import path_exist
from utils.core import pad_to_height

from dataset import get_meanpose
from model import get_autoencoder

from utils.motion import (
    preprocess_motion2d,
    postprocess_motion2d,
    openpose2motion,
    prebundle_openpose_to_motion,
)

from common import config
from utils.video import *

# Load to Openpose bundles (collected numpy matrices)

# Use the same squeeze as in predict.py

# Encode input video OpenPose bundles (preprocessed for LCM/Transmomo) into character-agnostic encoder

# Run DTW on the encoder over the whole video
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("vid1_kp", type=path_exist)
    parser.add_argument("vid2_kp", type=path_exist)

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
        type=int,
        required=True,
        help="video 1 shape: [H, W]",
    )

    parser.add_argument(
        "-v2s",
        "--v2-shape",
        nargs=2,
        metavar=("height", "width"),
        type=int,
        required=True,
        help="video 2 shape: [H, W]",
    )

    parser.add_argument("--vid1", type=path_exist, default=None, help="Path to video1 for alignment viewing")
    parser.add_argument("--vid2", type=path_exist, default=None, help="Path to video2 for alignment viewing")

    parser.add_argument('-o', '--out_dir', type=path_exist, default="./outputs", help="output saving directory")
    parser.add_argument("-ofn", "--out_fname", type=str, default=None, help="Output filename for generated video. Random otherwise")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    # fmt: on
    args = parser.parse_args()

    config.initialize(args)

    return args, config


def video_dtw(args, config):

    try:

        # mean/std pose
        mean_pose, std_pose = get_meanpose(config)

        # resize input
        h1, w1, scale1 = pad_to_height(config.img_size[0], *args.v1_shape)
        h2, w2, scale2 = pad_to_height(config.img_size[0], *args.v2_shape)

        input1, input2 = [
            prebundle_openpose_to_motion(path, scale=scale)
            if os.path.isfile(path)
            else openpose2motion(path, scale=scale)
            for path, scale in [(args.vid1_kp, scale1), (args.vid2_kp, scale2)]
        ]

        input1 = preprocess_motion2d(input1, mean_pose, std_pose)
        input2 = preprocess_motion2d(input2, mean_pose, std_pose)
        input2 = input2.to(config.device)

        # load trained model
        net = get_autoencoder(config)
        net.load_state_dict(torch.load(args.model_path))
        net.to(config.device)
        net.eval()

        with torch.no_grad():
            # Autoencode motion
            z1, z2 = net.mot_encoder(input1), net.mot_encoder(input2)

        # Dead-simple Euclidean cost matrix in the motion embedding (static appearance agnostic)
        cost = cdist(
            z1.squeeze(dim=0).detach().numpy().T,
            z2.squeeze(dim=0).detach().numpy().T,
            metric="euclidean",
        )
        alignment = dtw.dtw(
            x=cost,
            step_pattern="asymmetric",
            keep_internals=True,
            open_begin=True,
            open_end=True,
        )

        # Optional split-screen video view
        if args.vid1 and args.vid2:

            vid1, vid2 = [
                load_video_frames_to_npy(path)
                if os.path.isdir(path)
                else load_video_to_npy(path)
                for path in [args.vid1, args.vid2]
            ]

            out_fname = (
                args.out_fname
                if args.out_fname
                else "".join(random.choices(string.ascii_letters + string.digits, k=8))
                + ".mp4"
            )

            out_path = os.path.join(args.out_dir, out_fname)
            print("Saving video to file: {}".format(out_fname))

            write_video_to_file(
                align_with_interp_fill(
                    vid1, vid2, z1, z2, alignment, mean_pose, std_pose, net
                ),
                out_path,
            )

            # write_video_to_file(align_and_split_screen(vid1, vid2, alignment), out_path)

    except:
        print("Unable to render keypoint motion as video!")
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


if __name__ == "__main__":

    args, config = parse_args()
    video_dtw(args, config)
