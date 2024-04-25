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
| `inference.sh` | Runs inference using a default `batch_size=1` for the specified precision (fp32, bfloat16, fp16, int8). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `accuracy.sh` | Measures the model accuracy for the specified precision (fp32, bfloat16, fp16, int8). Only batch size 1 is supported for accuracy.|
| `inference_realtime_multi_instance.sh` | Runs realtime inference with `batch_size=1` for the specified precision (fp32, bfloat16, fp16, int8). |
| `inference_throughput_multi_instance.sh` | Runs multi-instance throughput with specified batch size and specified precision (fp32, bfloat16, fp16, int8). |


<!--- 50. Baremetal -->
### Pre-requisites:
- Creating a virtual environment is recommended:
    ```
    python3 -m venv tf-venv
    source tf-venv/bin/activate
    ```
- Install [Tensorflow](https://pypi.org/project/tf-nightly/)
- Clone the AI Reference Models repository:
    ```
    git clone https://github.com/IntelAI/models.git
    cd models
    ```
- Install model requirements:
    ```
    pip install -r models/object_detection/tensorflow/yolov5/inference/requirements.txt
    ```
### Pre-Trained Model

To get a TensorFlow pretrained model, you need to export it from a PyTorch model. Clone the [Ultralytics yolov5 repository](https://github.com/ultralytics/yolov5.git).
Generate the pretrained PyTorch model and then export to a Tensorflow supported format with the following commands:
```
python yolov5/models/tf.py --weights yolov5/yolov5s.pt
python yolov5/export.py --weights yolov5/yolov5s.pt --include pb

# Point the pretrained model to `PRETRAINED_MODEL` environment variable:
export PRETRAINED_MODEL=$(pwd)/yolov5/yolov5s.pb
```

### Run on Linux

Set environment variables to specify the dataset directory, precision to run, path to pretrained files and an output directory.
```
# Navigate to the models directory
cd models

# Set the required environment vars:
export PRECISION=<specify the precision to run: fp32, bfloat16, fp16, int8>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

Run the script:
./quickstart/object_detection/tensorflow/yolov5/inference/cpu/<script_name.sh>
```
