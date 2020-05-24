
import imageio
import ffmpeg
import dtw
import sys
import traceback
import pdb
import os

import numpy as np

from utils.core import pad_to_height


def load_video_frames_to_npy(path: str) -> np.ndarray:
    """Loads a video into a numpy array: [T, height, width, channels]"""
    try:
        # Temporary video file preserving exact frame numbers
        (
            ffmpeg
            .input("{}/*.png".format(path), pattern_type='glob')
            .output("tmp.mp4")
            .run()
        )

        probe = ffmpeg.probe("tmp.mp4")
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        out, _ = (
            ffmpeg
            .input("tmp.mp4")
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True)
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        os.remove("tmp.mp4")
        return video
    
    except:
        print("Unable to load video: {}".format(path))
        typ, value, tb = sys.exc_info()
        traceback.print_exc()

def save_video_to_file(vid: np.ndarray, path: str) -> None:
    """"""
    pass


def apply_alignment(vid1: np.ndarray, index: np.ndarray) -> np.ndarray:
    """apply alignment vid1->vid2. [T, height, width, channels]"""
    return np.take(vid1, index, axis=0)
    # if vid1.shape[0] < vid2.shape[0]:
    #     return np.take(vid1, align.index1, axis=0)
    # else:
    #     return np.take(vid2, align.index2, axis=0)

def make_split_screen(vid1: np.ndarray, vid2: np.ndarray) -> np.ndarray:
    """Stack videos in a vertical split-screen view. [T, height, width, channels]"""

    assert vid1.shape[0] == vid2.shape[0]  # videos same duration

    T = vid1.shape[0]
    min_w = min(vid1.shape[2], vid2.shape[2])
    max_w = max(vid1.shape[2], vid2.shape[2])

    if min_w == max_w:
        return np.concatenate([vid1, vid2], axis=1)
    else:

        h1, w1 = vid1.shape[1:3]
        h2, w2 = vid2.shape[1:3]

        left_pad = (max_w - min_w) // 2
        vert_split = np.empty((h1 + h2, max_w, T))

        if w1 < w2:
            vert_split[:h1, left_pad : (left_pad + w1)] = vid1
            vert_split[h1:, :] = vid2
        else:
            vert_split[:h2, left_pad : (left_pad + w2)] = vid2
            vert_split[h2:, :] = vid1

        return vert_split

def align_and_split_screen(vid1: np.ndarray, vid2: np.ndarray, align: dtw.DTW) -> np.ndarray:

    pdb.set_trace()

    vid_12 = apply_alignment(vid1, align.index1) 
    vid_21 = apply_alignment(vid2, align.index2) 


    pass

    # vid_aligned = apply_alignment(vid1, vid2, align)

    # splitscreen = videos_to_split_screen()
