#!/bin/bash
set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

IN_DIR=$1
OUT_DIR=$2
echo "Frame directory: $PWD/$IN_DIR"
echo "Output directory: $PWD/$OUT_DIR"

mkdir -p $OUT_DIR/render
mkdir -p $OUT_DIR/annotations

# Spin up DensePose container named 'dense'
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

# DensePose container must be named 'dense' for this to work
DENSEPOSE_CONTAINER_ID=$(docker ps -aqf "name=dense")
echo "Spun up DensePose: $DENSEPOSE_CONTAINER_ID"

# FIXME: This should work for multiple images.
nvidia-docker exec -it $DENSEPOSE_CONTAINER_ID \
  python2 tools/infer_multi.py \
      --im-dir /data \
      --output-dir /out \
      wts/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml
      # wts/DensePose_ResNet101_FPN_s1x-e2e.pkl configs/DensePose_ResNet101_FPN_s1x-e2e.yaml

# # TODO: Check that this actually works
# nvidia-docker exec -it $DENSEPOSE_CONTAINER_ID \
#   mv /densepose/test_vis.pkl /out/render ; \
#   mv /out/render/test_vis.pkl /out/render/frame_000001_annot.pkl

# https://github.com/facebookresearch/DensePose/blob/master/MODEL_ZOO.md  
      
# Kill the OpenPose Container
docker kill $DENSEPOSE_CONTAINER_ID
docker container rm $DENSEPOSE_CONTAINER_ID    