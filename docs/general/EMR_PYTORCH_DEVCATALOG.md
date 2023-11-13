# Optimize Intel® AI Reference Model Workloads with PyTorch* using Docker Containers

This document provides links to step-by-step instructions on how to leverage Reference model docker containers to run optimized open-source Deep Learning Training and Inference workloads using PyTorch* framework on 5th Generation Intel® Xeon® Scalable processors
## Use cases

The tables below provide links to run each use case using docker containers. The model scripts run on Linux. 

### Image Recognition

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| --------------------- |
| [ResNet 50](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/pytorch/resnet50/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Training | ImageNet 2012 |
| [ResNet 50](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/pytorch/resnet50/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | ImageNet 2012 |
| [ResNext-32x16d](https://github.com/IntelAI/models/blob/r3.1/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | ImageNet 2012 |

### Object Detection

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------|  ---------------------- |
| [Mask R-CNN](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/pytorch/maskrcnn/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Training | COCO 2017 |
| [Mask R-CNN](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Inference | COCO 2017 |
| [SSD-ResNet34](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Training | COCO 2017 |
| [SSD-ResNet34](https://github.com/IntelAI/models/blob/r3.1/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | COCO 2017 |

### Language Modeling 

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ---------------------- |
| [BERT large](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/pytorch/bert_large/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Training | Preprocessed Text dataset |
| [BERT large](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/pytorch/bert_large/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | SQuAD1.0 |
| [RNN-T](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/pytorch/rnnt/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | LibriSpeech |
| [RNN-T](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/pytorch/rnnt/training/cpu/DEVCATALOG.md) | FP32,BF32,FP16 Training | LibriSpeech |
| [DistilBERT base](https://github.com/IntelAI/models/blob/r3.1/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8-BF16,INT8-BF32 Inference | SST-2 |

### Recommendation 

| Model                                                  | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------|---------------------- |
| [DLRM](https://github.com/IntelAI/models/blob/r3.1/quickstart/recommendation/pytorch/dlrm/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 Training | Criteo Terabyte |
| [DLRM](https://github.com/IntelAI/models/blob/r3.1/quickstart/recommendation/pytorch/dlrm/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 Inference | Criteo Terabyte |
