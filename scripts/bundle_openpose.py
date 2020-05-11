import argparse
import json
import os
import pdb

import numpy as np

# TransMoMo only uses the first 15 keypoints on BODYPOSE_25
TRANSMOMO_KEYPOINT_STOP = 15
# Keypoints are (x,y)
KEYPOINT_DIM = 2


def bundle_keypoints(args):

    kp_files = [
        os.path.join(args.keypoint_dir, f)
        for f in os.listdir(args.keypoint_dir)
        if os.path.isfile(os.path.join(args.keypoint_dir, f)) and f.endswith(".json")
    ]

    kp_matrix = np.empty((TRANSMOMO_KEYPOINT_STOP, KEYPOINT_DIM, len(kp_files)))

    for f_index, kpf in enumerate(kp_files):
        with open(kpf) as f:
            payload = json.load(f)

            coords = [
                v
                for index, v in enumerate(payload["people"][0]["pose_keypoints_2d"])
                if (index + 1) % 3 != 0
            ][: (TRANSMOMO_KEYPOINT_STOP * KEYPOINT_DIM)]

            kp_matrix[:, 0, f_index] = coords[0::2]
            kp_matrix[:, 1, f_index] = coords[1::2]

    np.save(os.path.join(args.keypoint_dir, args.output_fname), kp_matrix)


if __name__ == "__main__":

    # Input folder path
    parser = argparse.ArgumentParser(description="Bundle OpenPose keypoint files")

    parser.add_argument("--keypoint-dir", dest="keypoint_dir", default=None, type=str)
    parser.add_argument("--output-fname", dest="output_fname", default=None, type=str)

    args = parser.parse_args()

    bundle_keypoints(args)
