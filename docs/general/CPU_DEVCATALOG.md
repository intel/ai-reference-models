# Latest Intel® AI Reference Model Optimizations for Intel® Xeon Scalable Processors

This document provides links to step-by-step instructions on how to leverage the latest reference model docker containers to run optimized open-source Deep Learning Training and Inference workloads using PyTorch* and TensorFlow* frameworks on Intel® Xeon® Scalable processors.

Note: The containers below are finely tuned to demonstrate best performance on Intel® Extension for PyTorch* and Intel® Optimized TensorFlow*  and are not intended for use in production. 

## Use cases

The tables below link to documentation on how to run each use case using docker containers. These containers were validated on a host running Linux. 

### Generative AI
| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [GPT-J](../../quickstart/language_modeling/pytorch/gptj/inference/cpu/CONTAINER.md) | FP32,BF16,FP16,INT8-FP32 | Inference | LAMBADA |
| PyTorch | [Llama 2](../../quickstart/language_modeling/pytorch/gptj/inference/cpu/CONTAINER.md) 7B,13B | FP32,BF16,FP16,INT8-FP32 | Inference | LAMBADA |
### Image Recognition

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [ResNet 50](../../quickstart/image_recognition/pytorch/resnet50/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Training | ImageNet 2012 |
| PyTorch | [ResNet 50](../../quickstart/image_recognition/pytorch/resnet50/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 | Inference | ImageNet 2012 |
| PyTorch | [ResNext-32x16d](../../quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 |Inference | ImageNet 2012 |
| TensorFlow | [ResNet 50v1.5](../../quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/README_DEV_CAT.md) | FP32,BF16,FP16 | Training | ImageNet 2012 |
| TensorFlow | [ResNet 50v1.5](../../quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 | Inference | ImageNet 2012 |
| TensorFlow | [MobileNet V1*](../../quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/README_DEV_CAT.md) | FP32,BF32,FP16,INT8 | Inference | ImageNet 2012 |

## Image Segmentation

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| TensorFlow | [3D U-Net MLPerf*](../../quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 Inference | BRATS 2019 |

### Object Detection

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch |[Mask R-CNN](../../quickstart/object_detection/pytorch/maskrcnn/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Training | COCO 2017 |
| PyTorch |[Mask R-CNN](../../quickstart/object_detection/pytorch/maskrcnn/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Inference | COCO 2017 |
| PyTorch |[SSD-ResNet34](../../quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Training | COCO 2017 |
| PyTorch |[SSD-ResNet34](../../quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 | Inference | COCO 2017 |
| PyTorch | [YOLO v7]() | FP32,BF16,FP16,INT8 | Inference | COCO 2017 |
| TensorFlow | [SSD-ResNet34](../../quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 |Training | COCO 2017 |
| TensorFlow | [SSD-ResNet34](../../quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 |Inference | COCO 2017  |
| TensorFlow | [SSD-MobileNet*](../../quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 | Inference | COCO 2017 |

### Language Modeling 

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [BERT large](../../quickstart/language_modeling/pytorch/bert_large/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Training | Preprocessed Text dataset |
| PyTorch |[BERT large](../../quickstart/language_modeling/pytorch/bert_large/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 | Inference | SQuAD1.0 |
| PyTorch | [RNN-T](../../quickstart/language_modeling/pytorch/rnnt/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 | Inference | LibriSpeech |
| PyTorch |[RNN-T](../../quickstart/language_modeling/pytorch/rnnt/training/cpu/DEVCATALOG.md) | FP32,BF32,FP16 | Training | LibriSpeech |
| PyTorch |[DistilBERT base](../../quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8-BF16,INT8-BF32 | Inference | SST-2 |
| TensorFlow | [BERT large](../../quickstart/language_modeling/tensorflow/bert_large/training/cpu/README_DEV_CAT.md) | FP32,BF16 | Training |  SQuAD and MRPC |
| TensorFlow | [BERT large](../../quickstart/language_modeling/tensorflow/bert_large/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 |Inference | SQuAD |
| TensorFlow | [DistilBERT Base](../../quickstart/language_modeling/tensorflow/distilbert_base/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 | Inference | SST-2 | 

## Language Translation
| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| TensorFlow | [Transformer_LT_mlperf*](../../quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/README_DEV_CAT.md) | FP32,BF16 | Training | WMT English-German dataset |
| TensorFlow | [Transformer_LT_mlperf*](../../quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 | Inference |  WMT English-German dataset |

### Recommendation 

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [DLRM](../../quickstart/recommendation/pytorch/dlrm/training/cpu/DEVCATALOG.md) | FP32,BF32,BF16 | Training | Criteo Terabyte |
| PyTorch | [DLRM](../../quickstart/recommendation/pytorch/dlrm/inference/cpu/DEVCATALOG.md) | FP32,BF32,BF16,INT8 | Inference | Criteo Terabyte |
| PyTorch | [DLRM v2]() | FP32,BF16,FP16,INT8 | Inference | Criteo Terabyte |
| TensorFlow | [DIEN](../../quickstart/recommendation/tensorflow/dien/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 | Inference | DIEN dataset |
