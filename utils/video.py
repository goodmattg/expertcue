import imageio_ffmpeg
import imageio
import dtw

import numpy as np


def load_video_to_npy(path: str) -> np.ndarray:
    """"""
    pass


def save_video_to_file(vid: np.ndarray, path: str) -> None:
    """"""
    pass


def apply_alignment(vid1: np.ndarray, vid2: np.ndarray, align: dtw.DTW) -> np.ndarray:
    """apply alignment vid1->vid2"""
    t1, t2 = vid1.shape[-1], vid2.shape[-1]

    if t1 < t2:
        dilated = np.empty_like(vid2)
        dilated = vid1[:, :, align.index1]
    else:
        dilated = np.empty_like(vid1)
        dilated = vid2[:, :, align.index2]

    return dilated


def videos_to_split_screen(vid1: np.ndarray, vid2: np.ndarray) -> np.ndarray:
    """"""

    assert vid1.shape[-1] == vid2.shape[-1]  # videos same duration

    pass
