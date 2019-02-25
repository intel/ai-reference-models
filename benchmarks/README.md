# Benchmark scripts

Training and inference scripts with Intel-optimized MKL

## Prerequisites

The benchmarking scripts can be run on Linux and require the following
dependencies to be installed:
* [Docker](https://docs.docker.com/install/)
* [Python](https://www.python.org/downloads/) 2.7 or later
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* `wget` for downloading pre-trained models

## Use Cases

| Use Case               | Framework     | Model               | Mode      | Benchmarking Instructions    |
| -----------------------| --------------| ------------------- | --------- |------------------------------|
| Adversarial Networks   | TensorFlow    | DCGAN               | Inference | [FP32](adversarial_networks/tensorflow/dcgan/README.md#fp32-inference-instructions) |
| Content Creation       | TensorFlow    | DRAW                | Inference | [FP32](content_creation/tensorflow/draw/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | Inception ResNet V2 | Inference | [FP32](image_recognition/tensorflow/inception_resnet_v2/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | Inception V3        | Inference | [Int8](image_recognition/tensorflow/inceptionv3/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv3/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | MobileNet V1        | Inference | [FP32](image_recognition/tensorflow/mobilenet_v1/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | ResNet 101          | Inference | [Int8](image_recognition/tensorflow/resnet101/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet101/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | ResNet 50           | Inference | [Int8](image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | SqueezeNet          | Inference | [FP32](image_recognition/tensorflow/squeezenet/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | Mask R-CNN          | Inference | [FP32](image_segmentation/tensorflow/maskrcnn/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | UNet                | Inference | [FP32](image_segmentation/tensorflow/unet/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | Transformer Language| Inference | [FP32](language_translation/tensorflow/transformer_language/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | Fast R-CNN          | Inference | [Int8](object_detection/tensorflow/fastrcnn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/fastrcnn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | R-FCN               | Inference | [FP32](object_detection/tensorflow/rfcn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | SSD-MobileNet       | Inference | [FP32](object_detection/tensorflow/ssd-mobilenet/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | NCF                 | Inference | [FP32](recommendation/tensorflow/ncf/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | Wide & Deep Large Dataset	| Inference | [Int8](recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions) [FP32](recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | Wide & Deep         | Inference | [FP32](recommendation/tensorflow/wide_deep/README.md#fp32-inference-instructions) |
| Text-to-Speech         | TensorFlow    | WaveNet             | Inference | [FP32](text_to_speech/tensorflow/wavenet/README.md#fp32-inference-instructions) |
