<!--- 0. Title -->
# Vision Transformer Inference

<!-- 10. Description -->

### Description
This document has instructions for running Vision Transformer inference for FP32. The model based on this [paper](https://arxiv.org/abs/2010.11929). Inference is performed for the task of image recognition. 
The following Hugging Face [pretrained-model](https://huggingface.co/google/vit-base-patch16-224) is used.

### Dataset

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running Vision Transformer.

```
Download Frozen graph for FP32, BF16 and FP16:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_11_0/HF-ViT-Base16-Img224-frozen.pb

```

```
Download Frozen graph for INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/nc_vit_int8_newapi.pb

```

## Run the model

### Run on Linux
```
# cd to your AI Reference Models directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the ImageNet TF records>
export PRECISION=<set the precision to "fp32", "bfloat16" or "fp16">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```

### Inference

#### Throughput
Default Batch Size 32
```
./quickstart/image_recognition/tensorflow/vision_transformer/inference/cpu/inference_throughput_multi_instance.sh
```

#### Latency
Default Batch Size 1
```
./quickstart/image_recognition/tensorflow/vision_transformer/inference/cpu/inference_realtime_multi_instance.sh
```

### Accuracy
```
./quickstart/image_recognition/tensorflow/vision_transformer/inference/cpu/accuracy.sh
```


