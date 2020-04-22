#!/bin/bash
set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

IN_DIR=$1
OUT_DIR=$2
echo "Frame directory: $PWD/$IN_DIR"
echo "Output directory: $PWD/$OUT_DIR"

mkdir -p $OUT_DIR/render
mkdir -p $OUT_DIR/keypoints

# Spin up OpenPose container named 'pose'
docker run -d \
  -it \
  --name pose \
  --net=host \
  -e DISPLAY \
  --runtime=nvidia \
  --mount type=bind,source=$PWD/$IN_DIR,target=/data \
  --mount type=bind,source=$PWD/$OUT_DIR,target=/out \
  exsidius/openpose:openpose

# OpenPose container must be named 'pose' for this to work
OPENPOSE_CONTAINER_ID=$(docker ps -aqf "name=pose")
echo "Spun up OpenPose: $OPENPOSE_CONTAINER_ID"

docker exec -it $OPENPOSE_CONTAINER_ID \
    ./build/examples/openpose/openpose.bin \
    --image_dir /data \
    --model_pose BODY_25 \
    --display 0 \
    --render_pose 2 \
    --hand \
    --hand_render 2 \
    --disable_blending \
    --write_images /out/render \
    --write_json /out/keypoints

# Kill the OpenPose Container
docker kill $OPENPOSE_CONTAINER_ID
docker container rm $OPENPOSE_CONTAINER_ID