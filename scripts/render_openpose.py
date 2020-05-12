import argparse
import numpy as np

from transmomo.lib.util.visualization import motion2video, motion2video_np, hex2rgb


def render_keypoints(args):

    # color = "#a50b69#b73b87#db9dc3"
    # color = np.array(hex2rgb(color))
    # print(color)

    # with open(args.input_fname, "rb") as f:
    #     motion = np.load(f)

    #     motion2video(motion, 512, 512, "blah.mp4", np.array([0.5, 0.5, 0.5]))
    return


if __name__ == "__main__":

    print(__name__)

    # Input folder path
    parser = argparse.ArgumentParser(
        description="Render OpenPose keypoints in TransMoMo format"
    )

    parser.add_argument("--input-fname", dest="input_fname", default=None, type=str)

    args = parser.parse_args()

    render_keypoints(args)
