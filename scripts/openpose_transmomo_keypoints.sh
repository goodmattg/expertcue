#!/bin/bash
set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

IN_VIDEO_PATH=$1
IN_VIDEO=$(basename $IN_VIDEO_PATH)

echo "Generating keypoints for: $IN_VIDEO"

# Spin up OpenPose container named 'pose'
docker run -d \
  -it \
  --name pose \
  --net=host \
  -e DISPLAY \
  --runtime=nvidia \
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
    --render_pose 0 \
    --write_json /

# Kill the OpenPose Container
# docker kill $OPENPOSE_CONTAINER_ID
# docker container rm $OPENPOSE_CONTAINER_ID

# # Create a tar bundle of the output
# tar -czvf openpose_frames_keypoints.tar.gz $OUT_DIR/render $OUT_DIR/keypoints
# mv openpose_frames_keypoints.tar.gz $OUT_DIR