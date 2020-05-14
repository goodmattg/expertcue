ffmpeg -i raw/scarecrow.mp4 -vf fps=60 frames/scarecrow/frame_%06d.png

# Start on OpenPose container with Data folder and CueNet mounted

docker run -d \
  -it \
  --name pose \
  --net=host \
  -e DISPLAY \
  --runtime=nvidia \
  --mount type=bind,source=/home/goodmanm/expertcue/data,target=/data \
  --mount type=bind,source=/home/goodmanm/expertcue,target=/expertcue \
  exsidius/openpose:openpose

## Keypoints from image frames

./build/examples/openpose/openpose.bin --image_dir _frames_ --model_pose BODY_25 --display 0 --render_pose 2 --hand --hand_render 2 --disable_blending --write_images _outdir_ --write_json _outdir_



python infer_pair.py  \
--config configs/transmomo.yaml \
--checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt \
--source airsquat.npy \
--target airsquat_c.npy  \
--source_width 1920 --source_height 1080 \
--target_height 1280 --target_width 720