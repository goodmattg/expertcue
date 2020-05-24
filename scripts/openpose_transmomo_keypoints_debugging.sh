#!/bin/bash
set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

IN_VIDEO_PATH=$1
IN_VIDEO=$(basename $IN_VIDEO_PATH)
OUT_PATH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)

OUT_BASENAME=${IN_VIDEO%.*}
OUT_KEYPOINT_NPY=${OUT_BASENAME}.npy
OUT_KEYPOINT_AVI=${OUT_BASENAME}_kp.avi

echo "Generating keypoints for: $IN_VIDEO"
echo "Temporarory store for keypoints: $OUT_PATH"
echo "Saving keypoints to: $OUT_KEYPOINT_NPY"

mkdir $PWD/$OUT_PATH

# Spin up OpenPose container named 'pose'
docker run -d \
  -it \
  --name pose \
  --net=host \
  -e DISPLAY \
  --runtime=nvidia \
  --mount type=bind,source=$PWD/$OUT_PATH,target=/out \
  exsidius/openpose:openpose

docker cp $IN_VIDEO_PATH pose:/$IN_VIDEO

# OpenPose container must be named 'pose' for this to work
OPENPOSE_CONTAINER_ID=$(docker ps -aqf "name=pose")
echo "Spun up OpenPose: $OPENPOSE_CONTAINER_ID"

docker exec -it $OPENPOSE_CONTAINER_ID \
    ./build/examples/openpose/openpose.bin \
    --video /$IN_VIDEO \
    --model_pose BODY_25 \
    --display 0 \
    --write_video /out/$OUT_KEYPOINT_AVI \
    --write_json /out \
    --write_images /out

# Kill and remove the OpenPose Container
docker kill $OPENPOSE_CONTAINER_ID
docker container rm $OPENPOSE_CONTAINER_ID

# Bundle OpenPose keypoints to TransMoMo format (15, 2, seq_length) .npy file
python scripts/bundle_openpose.py \
  --keypoint-dir $PWD/$OUT_PATH \
  --output-fname $OUT_KEYPOINT_NPY

# mv $PWD/$OUT_PATH/$OUT_KEYPOINT_NPY .
# mv $PWD/$OUT_PATH/$OUT_KEYPOINT_AVI .
# rm -rf $PWD/$OUT_PATH
