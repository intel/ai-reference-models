<!--- 0. Title -->
# TensorFlow Yolo V5 inference

<!-- 10. Description -->
## Description

This document has instructions for running YOLOv5 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in the
Yolo V5 quickstart scripts. The scripts require that the dataset
has been converted to the TF records format. See the
[COCO dataset](/datasets/coco/README.md) for instructions on downloading
and preprocessing the COCO validation dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision. To run inference for throughtput, set `BATCH_SIZE` environment variable. |

<!--- 50. Baremetal -->

### Pre-Trained Model

To get a TensorFlow pretrained model, you need to export it from a PyTorch model. Clone the [Ultralytics yolov5 repository](https://github.com/ultralytics/yolov5.git).
Install the required dependencies listed in [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt).
Generate the pretrained PyTorch model and then export to a Tensorflow supported format with the following commands:
```
python models/tf.py --weights yolov5s.pt
python export.py --weights yolov5s.pt --include pb
```

### Run on Linux

Set environment variables to specify the dataset directory, precision to run, path to pretrained files and an output directory.
```
# Navigate to the models directory
cd models

# Set the required environment vars:
export PRECISION=<specify the precision to run: fp32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

Run the script:
./quickstart/object_detection/tensorflow/yolov5/inference/cpu/<script_name.sh>
```
