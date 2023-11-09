# Optimize Intel® AI Reference Model Workloads with TensorFlow* using Docker Containers

This document provides links to step-by-step instructions on how to leverage Reference model docker containers to run optimized open-source Deep Learning Training and Inference workloads using TensorFlow* framework on 5th Generation Intel® Xeon® Scalable processors

## Use cases

The tables below provide links to run each use case using docker containers. The model scripts run on Linux. 

### Image Recognition

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ---------------------- |
| [ResNet 50v1.5](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/README_DEV_CAT.md) | FP32,BF16,FP16 Training | ImageNet 2012 |
| [ResNet 50v1.5](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 Inference | ImageNet 2012 |
| [MobileNet V1*](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/README_DEV_CAT.md) | FP32,BF32,FP16,INT8 Inference | ImageNet 2012 |

### Image Segmentation

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------|  ---------------------- |
| [3D U-Net MLPerf*](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 Inference | BRATS 2019 |

### Object Detection 

| Model                                                  | Model Documentation      | Dataset |
| ------------------------------------------------------ | ----------| ---------------------- |
| [SSD-ResNet34](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 Training | COCO 2017 training dataset |
| [SSD-ResNet34](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 Inference | COCO 2017 validation dataset |
| [SSD-MobileNet*](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 Inference | COCO 2017 validation dataset |

### Language Modeling 

| Model                                                  | Model Documentation      |  Dataset |
| ------------------------------------------------------ | ----------|---------------------- |
| [BERT large](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/tensorflow/bert_large/training/cpu/README_DEV_CAT.md) | FP32,BF16 Training |  SQuAD and MRPC |
| [BERT large](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 Inference | SQuAD |

### Language Translation 

| Model                                                  | Model Documentation     | Dataset |
| ------------------------------------------------------ | ----------|---------------------- |
| [Transformer_LT_mlperf*](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/README_DEV_CAT.md) | FP32,BF16 Training | WMT English-German dataset |
|  [Transformer_LT_mlperf*](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 Inference |  WMT English-German dataset |

### Recommendation

| Model                                                  | Model Documentation      | Dataset |
| ------------------------------------------------------ | ----------| ---------------------- |
|[DIEN](https://github.com/IntelAI/models/blob/r3.1/quickstart/recommendation/tensorflow/dien/training/cpu/README_DEV_CAT.md) | FP32 Training | DIEN dataset |
| [DIEN](https://github.com/IntelAI/models/blob/r3.1/quickstart/recommendation/tensorflow/dien/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 Inference | DIEN dataset |
