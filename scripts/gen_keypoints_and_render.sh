#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR/render
mkdir -p $OUT_DIR/keypoints

./build/examples/openpose/openpose.bin --image_dir $IN_DIR --model_pose BODY_25 --display 0 --render_pose 2 --hand --hand_render 2 --disable_blending --write_images $OUT_DIR/render --write_json $OUT_DIR/keypoints