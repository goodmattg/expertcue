
## Setup instructions
### Clone & Install
```
git clone --recurse-submodules https://github.com/goodmattg/expertcue

# Install the TransMoMo fork submodule as a package 
pip install transmomo.pytorch
pip install -r transmomo.pytorch/requirements.txt
```

## Video Manipulations

### Generate and collect OpenPose keypoints
```
./scripts/openpose_transmomo_keypoints.sh input_video.mp4
```

### "Motion" rendering. Render bundled OpenPose keypoints as video (.mp4)
```
python scripts/render_openpose.py \
    keypoints.npy \
    --source-height 1080 \
    --source-width 1920 \
    --output-fname motion_video.mp4
```