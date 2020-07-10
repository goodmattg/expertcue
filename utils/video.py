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
from utils.align import interpolate_fill, find_runs, run_boundaries
from model.networks import AutoEncoder3x
from utils.motion import postprocess_motion2d
from utils.visualization import motion2video, hex2rgb

from typing import Tuple


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
            ffmpeg.input("{}/*.png".format(path), pattern_type="glob")
            .output("tmp.mp4")
            .run()
        )

        probe = ffmpeg.probe("tmp.mp4")
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        out, _ = (
            ffmpeg.input("tmp.mp4")
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        os.remove("tmp.mp4")
        return video

    except:
        print("Unable to load video: {}".format(path))
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


def write_video_to_file(
    vid: np.ndarray, path: str, framerate=25, vcodec="libx264"
) -> None:
    torchvision.io.write_video(
        path, torch.from_numpy(vid), framerate, video_codec=vcodec
    )


def save_video_to_file(
    vid: np.ndarray, path: str, framerate=25, vcodec="libx264"
) -> None:
    """"""
    n, height, width, channels = vid.shape
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height)
        )
        .output(path, pix_fmt="yuv420p", vcodec=vcodec, r=framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    vid = vid.astype(np.uint8)

    for ix in range(n):
        process.stdin.write(vid[ix].astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def apply_alignment(vid: np.ndarray, index: np.ndarray) -> np.ndarray:
    """apply alignment vid1->vid2. [T, height, width, channels]"""
    return np.take(vid, index, axis=0)


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


def align_and_split_screen(
    vid1: np.ndarray, vid2: np.ndarray, align: dtw.DTW
) -> np.ndarray:

    vid_12 = apply_alignment(vid1, align.index1)
    vid_21 = apply_alignment(vid2, align.index2)

    vert_split = make_split_screen(vid_12, vid_21)
    return vert_split


def align_with_interp_fill(
    motion1: np.ndarray,
    motion2: np.ndarray,
    align: dtw.DTW,
    mean_pose: np.ndarray,
    std_pose: np.ndarray,
    net: AutoEncoder3x,
    vid1_shape: Tuple[int, int],
    vid2_shape: Tuple[int, int],
    vid1: np.ndarray = None,
    vid2: np.ndarray = None,
) -> np.ndarray:

    if vid1 and vid2:
        # Videos with duplicated frames to create alignment
        print("Applying alignment to videos")
        vid_12 = apply_alignment(vid1, align.index1)
        vid_21 = apply_alignment(vid2, align.index2)

    # VIDEO 2
    h1, w1 = vid1_shape
    h2, w2 = vid2_shape

    # Find repeated frames ("runs") in the frame alignment
    _, starts, lengths = find_runs(align.index2)
    boundaries = run_boundaries(starts, lengths)

    # repeat_align_segs format is"
    # [Tuple(left_boundary_index, left_boundary_val), (Tuple(right_boundary_index, right_boundary_val)), [middle_indices]]

    repeat_align_segs = [
        (align.index2[start], align.index2[end], idx) for start, end, idx in boundaries
    ]

    # Only motion interpolate if any repeated frames
    if repeat_align_segs:

        # Original video latent embeddings
        # fmt: off
        with torch.no_grad():
            z_mot2 = net.mot_encoder(motion2)
            z_body2 = net.body_encoder(motion2[:, :-2, :]).repeat(1, 1, z_mot2.shape[-1])
            z_view2 = net.view_encoder(motion2[:, :-2, :]).repeat(1, 1, z_mot2.shape[-1])

            post = net.decoder(torch.cat([z_mot2, z_body2, z_view2], dim=1))
            out = postprocess_motion2d(post, mean_pose, std_pose, w2 // 2, h2 // 2)
            motion2video(
                out, h2, w2, "test1.mp4", hex2rgb("#a50b69#b73b87#db9dc3"), save_video=True
            )

        # fmt: on

        # Interpolation in motion embedding space for repeated segments
        # NOTE: This could be done in pure Torch without Numpy
        z_mot2_fills = interpolate_fill(
            repeat_align_segs, z_mot2.squeeze().numpy(), axis=-1
        )

        # Embeddings with alignment applied (repeated segments)
        z_mot2_hat = torch.index_select(z_mot2, -1, torch.from_numpy(align.index2))
        z_body2_hat = torch.index_select(z_body2, -1, torch.from_numpy(align.index2))
        z_view2_hat = torch.index_select(z_view2, -1, torch.from_numpy(align.index2))

        # Fill in the repeated segments in the aligned video with interpolated segments
        for (start, end, idx), interp_segment in zip(boundaries, z_mot2_fills):
            # pdb.set_trace()
            # print(start, end, idx)
            # if (
            #     torch.isnan(torch.from_numpy(interp_segment).T.unsqueeze(dim=0).clone())
            #     .any()
            #     .item()
            # ):
            #     print("Found a interp segment with NaNs")
            z_mot2_hat[:, :, idx] = (
                torch.from_numpy(interp_segment).T.unsqueeze(dim=0).float()
            )
            if torch.isnan(z_mot2_hat).any().item():
                print("The interpolated embedding has NaNs")

        # pdb.set_trace()

        # Go from interpolated motion embedding to skeleton image
        out = net.decoder(torch.cat([z_mot2_hat, z_body2_hat, z_view2_hat], dim=1))
        post = postprocess_motion2d(out, mean_pose, std_pose, w2 // 2, h2 // 2)

        temp_vid = motion2video(
            post, h2, w2, "test2.mp4", hex2rgb("#a50b69#b73b87#db9dc3"), save_video=True
        )

        pdb.set_trace()
        # for _, _, idx in boundaries:
        #     vid_21[align.index2[idx]] = temp_vid[align_index2[idx]]

        return out
        # Replace video frame with skeleton

    pass
