<!--- 0. Title -->
# TensorFlow Tiny Yolo V4 inference

<!-- 10. Description -->
## Description

This document has instructions for running TinyYoloV4 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets
The [COCO validation dataset](http://cocodataset.org) is used in the Tiny Yolo V4 quickstart scripts.
The scripts require that the dataset be resized and then converted to the TF records format.
See [COCO dataset](/datasets/coco/README.md) for instructions on downloading and preprocessing the COCO validation dataset.
The dataset can be resized with the script in [this](https://github.com/MSch8791/coco_dataset_resize.git) repository.
Note that you must resize the dataset before converting to the TF records format.

In summary:
1. Download COCO dataset
2. Resize the dataset
3. Convert to TF records

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision. To run inference for throughtput, set `BATCH_SIZE` environment variable. |

<!--- 50. Baremetal -->

### Pre-Trained Model

To get a TensorFlow pre-trained model, you need to download the weights file and then convert it to the frozen graph (.pb format). You can do this with the following steps:
```
git clone https://github.com/TNTWEN/OpenVINO-YOLOV4.git
cd OpenVINO-YOLOV4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny
```


### Run on Linux

Set environment variables to specify the dataset directory, precision to run, path to pretrained files and an output directory.
Install the required dependencies listed in [requirements.txt](/models/object_detection/tensorflow/tiny-yolov4/inference/requirements.txt).
```
# Navigate to the models directory
cd models

# Set the required environment vars:
export PRECISION=<specify the precision to run: fp32 or bfloat16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

Run the script:
./quickstart/object_detection/tensorflow/tiny-yolov4/inference/cpu/<script_name.sh>
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your AI Reference Models directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to the ImageNet TF records>
set PRECISION=<set the precision to fp32 or bfloat16>
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

# Run a quickstart script
bash quickstart\object_detection\tensorflow\tiny-yolov4\inference\cpu\<script name>.sh
```
