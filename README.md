
## Setup instructions
### Clone & Install
```
git clone --recurse-submodules https://github.com/goodmattg/expertcue

# Install the TransMoMo fork submodule as a package 
pip install -e transmomo.pytorch
pip install -r transmomo.pytorch/requirements.txt

# Follow TransMoMo instructions to download Mixamo data
mkdir -p transmomo.pytorch/transmomo/data/mixamo
mkdir -p transmomo.pytorch/transmomo/out
cd transmomo.pytorch/transmomo/data/mixamo
gdown --id 1z0kD_F4jHk2sMqgvYOPfTBsguU7uGY1x
unzip mixamo_36_800_24.zip
rm mixamo_36_800_24.zip
cd ../../
sh scripts/preprocess.sh

# Follow TransMoMo instructions to download pretrained model
gdown --id 120LeeR1WjdO0Emk_6hVRERu1I6Bimi6Q
unzip transmomo_mixamo_36_800_24.zip
rm transmomo_mixamo_36_800_24.zip
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