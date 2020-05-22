import argparse
import json
import os
import re

import numpy as np

# TransMoMo only uses the first 15 keypoints on BODYPOSE_25
TRANSMOMO_KEYPOINT_STOP = 15
# Keypoints are (x,y)
KEYPOINT_DIM = 2


def bundle_keypoints(args):

    # Keypoint file regex. OpenPose automatically append "_keypoints.json" to frame number
    KEYPOINT_REGEX = "{}_([0-9]+)_[a-zA-Z0-9]+.json".format(args.keypoint_base)

    kp_files = [
        f
        for f in os.listdir(args.keypoint_dir)
        if os.path.isfile(os.path.join(args.keypoint_dir, f))
        and re.match(KEYPOINT_REGEX, f) is not None
    ]

    sorted_files = [
        tup[1]
        for tup in sorted(
            [
                (
                    int(re.match(KEYPOINT_REGEX, f).group(1)),
                    os.path.join(args.keypoint_dir, f),
                )
                for f in kp_files
            ],
            key=lambda tup: tup[0],
        )
    ]

    kp_matrix = np.empty((TRANSMOMO_KEYPOINT_STOP, KEYPOINT_DIM, len(sorted_files)))

    # FIXME: This doesn't handle missing joint positions the same way LCM/TransMoMo do.
    # Use previous last available frame joint to fill in missing joints

    for f_index, kpf in enumerate(sorted_files):
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

    # fmt: off
    parser.add_argument("--keypoint-dir", dest="keypoint_dir", required=True, default=None, type=str)
    parser.add_argument("--keypoint-base", dest="keypoint_base", required=True, default=None, type=str)
    parser.add_argument("--output-fname", dest="output_fname", required=True, default=None, type=str)
    # fmt: on
    args = parser.parse_args()

    bundle_keypoints(args)
