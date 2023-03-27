# Model Zoo for Intel® Architecture Workloads Optimized for the Intel® Data Center GPU Flex Series

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/IntelAI/models/blob/master/quickstart/ipex-tool-container/gpu/devcatalog.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/IntelAI/models/blob/master/quickstart/tf-tool-container/gpu/devcatalog.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers are optimized for Linux*. 


| Model                            | Framework                  | Mode  |   Documentation |  Dataset |
| ----------------------------|     ---------- | ----------| ------------------- | ------------ |
| [ResNet 50 v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | TensorFlow | Inference| [INT8](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/devcatalog.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | Inference | [INT8 ](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/devcatalog.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [SSD-MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | Inference | [INT8](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/devcatalog.md) | [COCO 2017](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md#datasets)  |
| [YOLO v4](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | Inference |[INT8](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/yolov4/inference/gpu/devcatalog.md) | [COCO 2017](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md#datasets) |
| [SSD-MobileNet](https://arxiv.org/pdf/1704.04861.pdf) | TensorFlow | Inference | [INT8](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/gpu/devcatalog.md)| [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |

