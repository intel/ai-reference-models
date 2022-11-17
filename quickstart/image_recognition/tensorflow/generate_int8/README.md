## How to generate int8 model (ResNet50, ResNet101, Inception V3 and MobileNet V1)

Setup your environment using the instructions below
```shell
pip install intel-tensorflow'>=2.5.0'
pip install neural-compressor
git clone https://github.com/IntelAI/models.git
```
Download pre-trained fp32 model and export file name for later.
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet101-fp32-inference.tar.gz
export PRE_TRAINED_FP32_MODEL="$(pwd)/resnet101_fp32_pretrained_model.pb"

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb
export PRE_TRAINED_FP32_MODEL="$(pwd)/resnet50_fp32_pretrained_model.pb"

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenetv1_fp32_pretrained_model.pb
export PRE_TRAINED_FP32_MODEL="$(pwd)/mobilenetv1_fp32_pretrained_model.pb"

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb
export PRE_TRAINED_FP32_MODEL="$(pwd)/inceptionv3_fp32_pretrained_model.pb"
```
Switch to the directory
```shell
cd quickstart/image_recognition/tensorflow/generate_int8
```
Run
```shell
bash ./quantize.sh --topology=resnet101 --dataset_location=/PATH/TO/DATASET \
    --fp32_model=${PRE_TRAINED_FP32_MODEL} \
    --int8_model=./resnet-101-inc-int8-inference.pb

bash ./quantize.sh --topology=resnet50 --dataset_location=/PATH/TO/DATASET \
    --fp32_model=${PRE_TRAINED_FP32_MODEL} \
    --int8_model=./resnet-50-inc-int8-inference.pb

bash ./quantize.sh --topology=mobilenetv1 --dataset_location=/PATH/TO/DATASET \
    --fp32_model=${PRE_TRAINED_FP32_MODEL} \
    --int8_model=./mobilenet-v1-inc-int8-inference.pb
    
bash ./quantize.sh --topology=inceptionv3 --dataset_location=/PATH/TO/DATASET \
    --fp32_model=${PRE_TRAINED_FP32_MODEL} \
    --int8_model=./inception-v3-inc-int8-inference.pb
```