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
| Adversarial Networks   | TensorFlow    | [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)               | Inference | [FP32](adversarial_networks/tensorflow/dcgan/README.md#fp32-inference-instructions) |
| Content Creation       | TensorFlow    | [DRAW](https://arxiv.org/pdf/1502.04623.pdf)               | Inference | [FP32](content_creation/tensorflow/draw/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception ResNet V2](https://arxiv.org/pdf/1602.07261.pdf) | Inference | [FP32](image_recognition/tensorflow/inception_resnet_v2/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv3/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv3/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V4](https://arxiv.org/pdf/1602.07261.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv4/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv4/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [MobileNet V1](https://arxiv.org/pdf/1704.04861.pdf)        | Inference | [FP32](image_recognition/tensorflow/mobilenet_v1/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)          | Inference | [Int8](image_recognition/tensorflow/resnet101/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet101/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)           | Inference | [Int8](image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)          | Inference | [FP32](image_recognition/tensorflow/squeezenet/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf)            | Inference | [FP32](image_segmentation/tensorflow/3d_unet/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)          | Inference | [FP32](image_segmentation/tensorflow/maskrcnn/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | [UNet](https://arxiv.org/pdf/1505.04597.pdf)                | Inference | [FP32](image_segmentation/tensorflow/unet/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | [GNMT](https://arxiv.org/pdf/1609.08144.pdf)                | Inference | [FP32](language_translation/tensorflow/gnmt/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | [Transformer Language](https://arxiv.org/pdf/1706.03762.pdf)| Inference | [FP32](language_translation/tensorflow/transformer_language/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)               | Inference | [Int8](object_detection/tensorflow/rfcn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/rfcn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)        | Inference | [Int8](object_detection/tensorflow/faster_rcnn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/faster_rcnn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-MobileNet](https://arxiv.org/pdf/1704.04861.pdf)       | Inference | [FP32](object_detection/tensorflow/ssd-mobilenet/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [NCF](https://arxiv.org/pdf/1708.05031.pdf)                 | Inference | [FP32](recommendation/tensorflow/ncf/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf)	| Inference | [Int8](recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions) [FP32](recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf)         | Inference | [FP32](recommendation/tensorflow/wide_deep/README.md#fp32-inference-instructions) |
| Text-to-Speech         | TensorFlow    | [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)             | Inference | [FP32](text_to_speech/tensorflow/wavenet/README.md#fp32-inference-instructions) |
