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



F0422 15:32:44.672654    21 cudnn_conv_layer.cpp:53] Check failed: status == CUDNN_STATUS_SUCCESS (4 vs. 0)  CUDNN_STATUS_INTERNAL_ERROR
*** Check failure stack trace: ***
    @     0x7f26e6cbb5cd  google::LogMessage::Fail()
    @     0x7f26e6cbd433  google::LogMessage::SendToLog()
    @     0x7f26e6cbb15b  google::LogMessage::Flush()
    @     0x7f26e6cbde1e  google::LogMessageFatal::~LogMessageFatal()
    @     0x7f26e6233613  caffe::CuDNNConvolutionLayer<>::LayerSetUp()
    @     0x7f26e62acb53  caffe::Net<>::Init()
    @     0x7f26e62af990  caffe::Net<>::Net()
    @     0x7f26e81f84e8  op::NetCaffe::initializationOnThread()
    @     0x7f26e81661fe  op::addCaffeNetOnThread()
    @     0x7f26e81673f6  op::PoseExtractorCaffe::netInitializationOnThread()
    @     0x7f26e816c3d0  op::PoseExtractorNet::initializationOnThread()
    @     0x7f26e8162f51  op::PoseExtractor::initializationOnThread()
    @     0x7f26e815e411  op::WPoseExtractor<>::initializationOnThread()
    @     0x7f26e82104d1  op::Worker<>::initializationOnThreadNoException()
    @     0x7f26e8210610  op::SubThread<>::initializationOnThread()
    @     0x7f26e8213418  op::Thread<>::initializationOnThread()
    @     0x7f26e821361d  op::Thread<>::threadFunction()
    @     0x7f26e7ba0c80  (unknown)
    @     0x7f26e72f26ba  start_thread
    @     0x7f26e760f41d  clone
    @              (nil)  (unknown)
"docker exec -it $OPENPOSE_CONTAINER_ID ./build/examples/openpose/openpose.bin --image_dir /data --model_pose BODY_25 --display 0 --render_pose 2 --hand --hand_render 2 --disable_blending --write_images /out/render --write_json /out/keypoints" command filed with exit code 134.