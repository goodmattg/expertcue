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
nvidia-docker run -d \
  -it \
  --name dense \
  --net=host \
  -e DISPLAY \
  --runtime=nvidia \
  --mount type=bind,source=$PWD/$IN_DIR,target=/data \
  --mount type=bind,source=$PWD/$OUT_DIR,target=/out \
  --mount type=bind,source=$PWD/models/densepose_pretrained,target=/densepose/wts \
  garyfeng/densepose:latest

  # densepose:c2-cuda9-cudnn7-wtsdata2

# OpenPose container must be named 'pose' for this to work
DENSEPOSE_CONTAINER_ID=$(docker ps -aqf "name=dense")
echo "Spun up DensePose: $DENSEPOSE_CONTAINER_ID"

nvidia-docker exec -it $DENSEPOSE_CONTAINER_ID \
  python2 tools/infer.py \
      --im /data/frame_000001.png \
      --output-dir /out/render \
      wts/DensePose_ResNet101_FPN_s1x-e2e.pkl configs/DensePose_ResNet101_FPN_s1x-e2e.yaml

nvidia-docker exec -it $DENSEPOSE_CONTAINER_ID \
  mv /densepose/test_vis.pkl /out/render
      
# Kill the OpenPose Container
docker kill $DENSEPOSE_CONTAINER_ID
docker container rm $DENSEPOSE_CONTAINER_ID    