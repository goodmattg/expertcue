
import imageio
import ffmpeg
import dtw
import sys
import traceback
import pdb
import os
import torchvision
import torch

import numpy as np

from utils.core import pad_to_height

def load_video_to_npy(path: str) -> np.ndarray:
    try:
        vframes, _, _ = torchvision.io.read_video(path)
        return vframes.numpy()
    except: 
        print("Unable to load video: {}".format(path))
        typ, value, tb = sys.exc_info()
        traceback.print_exc()        

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

def write_video_to_file(vid: np.ndarray, path: str, framerate=25, vcodec="libx264") -> None:
    torchvision.io.write_video(path, torch.from_numpy(vid), framerate, video_codec=vcodec)

def save_video_to_file(vid: np.ndarray, path: str, framerate=25, vcodec="libx264") -> None:
    """"""
    n, height, width, channels = vid.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(path, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    vid = vid.astype(np.uint8)

    for ix in range(n):
        process.stdin.write(
            vid[ix]
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def apply_alignment(vid: np.ndarray, index: np.ndarray) -> np.ndarray:
    """apply alignment vid1->vid2. [T, height, width, channels]"""
    return np.take(vid, index, axis=0)
    # if vid1.shape[0] < vid2.shape[0]:
    #     return np.take(vid1, align.index1, axis=0)
    # else:
    #     return np.take(vid2, align.index2, axis=0)

def make_split_screen(vid1: np.ndarray, vid2: np.ndarray) -> np.ndarray:
    """Stack videos in a vertical split-screen view. [T, height, width, channels]"""

    assert vid1.shape[0] == vid2.shape[0]  # videos same duration

    T = vid1.shape[0]
    ch = vid1.shape[-1]
    min_w = min(vid1.shape[2], vid2.shape[2])
    max_w = max(vid1.shape[2], vid2.shape[2])

    if min_w == max_w:
        return np.concatenate([vid1, vid2], axis=1)
    else:
        h1, w1 = vid1.shape[1:3]
        h2, w2 = vid2.shape[1:3]

        left_pad = (max_w - min_w) // 2
        vert_split = np.empty((T, h1 + h2, max_w, ch))

        if w1 < w2:               
            vert_split[:, :h1, left_pad : (left_pad + w1)] = vid1
            vert_split[:, h1:, :] = vid2
        else:
            vert_split[:, :h2, left_pad : (left_pad + w2)] = vid2
            vert_split[:, h2:, :] = vid1

        return vert_split

def align_and_split_screen(vid1: np.ndarray, vid2: np.ndarray, align: dtw.DTW) -> np.ndarray:

    vid_12 = apply_alignment(vid1, align.index1) 
    vid_21 = apply_alignment(vid2, align.index2) 

    vert_split = make_split_screen(vid_12, vid_21)
    return vert_split

    # vid_aligned = apply_alignment(vid1, vid2, align)

    # splitscreen = videos_to_split_screen()
