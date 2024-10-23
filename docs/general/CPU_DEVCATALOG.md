# Latest Intel® AI Reference Model Optimizations for Intel® Xeon Scalable Processors

This document provides links to step-by-step instructions on how to leverage the latest reference model docker containers to run optimized open-source Deep Learning Training and Inference workloads using PyTorch* and TensorFlow* frameworks on Intel® Xeon® Scalable processors.

Note: The containers below are finely tuned to demonstrate best performance on Intel® Extension for PyTorch* and Intel® Optimized TensorFlow*  and are not intended for use in production. 

## Use cases

The tables below link to documentation on how to run each use case using docker containers. These containers were validated on a host running Linux. 

### Generative AI
| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [GPT-J](../../models_v2/pytorch/gptj/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,FP16,INT8-FP32 | Inference | LAMBADA |
| PyTorch | [Llama 2](../../models_v2/pytorch/llama/inference/cpu/CONTAINER.md) 7B,13B | FP32,BF32,BF16,FP16,INT8-FP32 | Inference | LAMBADA |
| PyTorch | [Llama 2](../../models_v2/pytorch/llama/training/cpu/CONTAINER.md) 7B | FP32,BF32,BF16,FP16 | Training | LAMBADA | 
| PyTorch | [ChatGLM](../../models_v2/pytorch/chatglm/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,FP16,INT8-FP32 | Inference | LAMBADA | 
| PyTorch | [LCM](../../models_v2/pytorch/LCM/inference/cpu/CONTAINER.md) |  FP32,BF32,BF16,FP16,INT8-FP32,INT8-BF16 | Inference | COCO 2017 |
| PyTorch | [Stable Diffusion](../../models_v2/pytorch/stable_diffusion/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,FP16,INT8-FP32,INT8-BF16 | Inference | COCO 2017 |

### Image Recognition

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [ResNet 50](../../models_v2/pytorch/resnet50/training/cpu/CONTAINER.md) | FP32,BF32,BF16 | Training | ImageNet 2012 |
| PyTorch | [ResNet 50](../../models_v2/pytorch/resnet50/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8 | Inference | ImageNet 2012 |
| PyTorch | [Vision Transformer](../../models_v2/pytorch/vit/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8-FP32,INT8-BF16 | Inference | ImageNet 2012 |
| TensorFlow | [MobileNet V1*](../../quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/README_DEV_CAT.md) | FP32,BF32,FP16,INT8 | Inference | ImageNet 2012 |

## Image Segmentation

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| TensorFlow | [3D U-Net MLPerf*](../../quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 | Inference | BRATS 2019 |

### Object Detection

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch |[Mask R-CNN](../../models_v2/pytorch/maskrcnn/training/cpu/CONTAINER.md) | FP32,BF32,BF16 | Training | COCO 2017 |
| PyTorch |[Mask R-CNN](../../models_v2/pytorch/maskrcnn/inference/cpu/CONTAINER.md) | FP32,BF32,BF16 | Inference | COCO 2017 |
| PyTorch |[SSD-ResNet34](../../models_v2/pytorch/ssd-resnet34/training/cpu/CONTAINER.md) | FP32,BF32,BF16 | Training | COCO 2017 |
| PyTorch |[SSD-ResNet34](../../models_v2/pytorch/ssd-resnet34/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8 | Inference | COCO 2017 |
| PyTorch |[YOLO v7](../../models_v2/pytorch/yolov7/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,FP16,INT8 | Inference | COCO 2017 |
| TensorFlow | [SSD-ResNet34](../../quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 |Training | COCO 2017 |
| TensorFlow | [SSD-ResNet34](../../quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/README_DEV_CAT.md) | FP32,BF16,INT8 |Inference | COCO 2017  |
| TensorFlow | [SSD-MobileNet*](../../quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16,INT8 | Inference | COCO 2017 |

### Language Modeling 

| Framework | Model                                                  | Precisions | Mode |  Dataset |
| --------| ------------------------------------------------------ | ---------- | ------| --------------------- |
| PyTorch | [BERT large](../../models_v2/pytorch/bert_large/training/cpu/CONTAINER.md) | FP32,BF32,BF16,FP16 | Training | Preprocessed Text dataset |
| PyTorch |[BERT large](../../models_v2/pytorch/bert_large/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8 | Inference | SQuAD1.0 |
| PyTorch | [RNN-T](../../models_v2/pytorch/rnnt/training/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8 | Inference | LibriSpeech |
| PyTorch |[RNN-T](../../models_v2/pytorch/rnnt/inference/cpu/CONTAINER.md) | FP32,BF32,FP16 | Training | LibriSpeech |
| PyTorch |[DistilBERT base](../../models_v2/pytorch/distilbert/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8-BF16,INT8-BF32 | Inference | SST-2 |
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
| PyTorch | [DLRM](../../models_v2/pytorch/dlrm/training/cpu/CONTAINER.md) | FP32,BF32,BF16 | Training | Criteo Terabyte |
| PyTorch | [DLRM](../../models_v2/pytorch/dlrm/inference/cpu/CONTAINER.md) | FP32,BF32,BF16,INT8 | Inference | Criteo Terabyte |
| PyTorch | [DLRM v2](../../models_v2/pytorch/torchrec_dlrm/inference/cpu/CONTAINER.md) | FP32,BF16,FP16,INT8 | Inference | Criteo Terabyte |
| TensorFlow | [DIEN](../../quickstart/recommendation/tensorflow/dien/inference/cpu/README_DEV_CAT.md) | FP32,BF32,BF16 | Inference | DIEN dataset |
