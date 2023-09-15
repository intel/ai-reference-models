# Model Zoo for Intel® Architecture Workloads Optimized for the Intel® Data Center GPU Flex Series

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/intel/intel-extension-for-pytorch/blob/v2.0.110%2Bxpu/docker/README.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/intel/intel-extension-for-tensorflow/blob/v2.13.0.0/docker/README.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers are optimized for Linux*. 


| Model                            | Framework                  | Mode and Documentation     |  Dataset |
| ----------------------------|     ---------- | ----------| ------------ |
| [ResNet 50 v1.5](https://github.com/tensorflow/models/tree/v2.12.1/official/legacy/image_classification/resnet) | TensorFlow | [INT8 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/DEVCATALOG_FLEX.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/v2.12.1/datasets/imagenet/README.md) |
| [MaskRCNN](https://arxiv.org/abs/1703.06870) | TensorFlow | [FP16 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/image_segmentation/tensorflow/maskrcnn/inference/gpu/DEVCATALOG.md) | [COCO 2017](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/image_segmentation/tensorflow/maskrcnn/inference/gpu/DEVCATALOG.md#download-dataset) |
| [EfficientNet](https://arxiv.org/abs/1905.11946) B0,B3 | TensorFlow| [FP16 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/image_recognition/tensorflow/efficientnet/inference/gpu/DEVCATALOG.md) | Dummy Image
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | TensorFlow | [FP32,FP16 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/generative-ai/tensorflow/stable_diffusion/inference/gpu/DEVCATALOG.md) | Text prompts |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [INT8 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/DEVCATALOG_FLEX.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/v2.12.1/datasets/imagenet/README.md) |
| [YOLOv5](https://ui.adsabs.harvard.edu/abs/2021zndo...4679653J/abstract) | PyTorch | [FP16 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/object_detection/pytorch/yolov5/inference/gpu/DEVCATALOG.md) | Dummy Image |
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | PyTorch | [FP32,FP16 Inference](https://github.com/IntelAI/models/blob/v2.12.1/quickstart/generative-ai/pytorch/stable_diffusion/inference/gpu/DEVCATALOG.md) | Text prompts
| [SSD-MobileNet](https://arxiv.org/pdf/1704.04861.pdf) | TensorFlow | [INT8 Inference](https://github.com/IntelAI/models/blob/v2.11.1/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/gpu/DEVCATALOG.md) | [COCO 2017](https://github.com/IntelAI/models/tree/v2.11.1/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | [INT8 Inference](https://github.com/IntelAI/models/blob/v2.11.1/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/DEVCATALOG.md) | [COCO 2017](https://github.com/IntelAI/models/blob/v2.11.1/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md#datasets)  |
| [YOLOv4](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | [INT8 Inference](https://github.com/IntelAI/models/blob/v2.11.1/quickstart/object_detection/pytorch/yolov4/inference/gpu/DEVCATALOG.md) | [COCO 2017](https://github.com/IntelAI/models/blob/v2.11.1/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md#datasets) |


**Note**: SSD-MobileNet and YOLOv4 models are supported on older Intel® Extension for TensorFlow* v2.12 and Intel® Extension for PyTorch* 1.13.100+xpu versions. The other models in the list are validated on Intel® Extension for TensorFlow* v2.13 and Intel® Extension for PyTorch* 2.0.100+xpu versions.
