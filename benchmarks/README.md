# Model Zoo Scripts

Training and inference scripts with Intel-optimized MKL

## Prerequisites

The model scripts can be run on Linux and require the following
dependencies to be installed:
* [Docker](https://docs.docker.com/install/) also support bare metal run
* [Python](https://www.python.org/downloads/) 3.5 or later
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* `wget` for downloading pre-trained models

## TensorFlow Use Cases

| Use Case               | Framework     | Model               | Mode      | Instructions    |
| -----------------------| --------------| ------------------- | --------- |------------------------------|
| Content Creation       | TensorFlow    | [DRAW](https://arxiv.org/pdf/1502.04623.pdf)               | Inference | [FP32](content_creation/tensorflow/draw/README.md#fp32-inference-instructions) |
| Face Detection and Alignment | TensorFlow    | [MTCC](https://arxiv.org/pdf/1604.02878.pdf)               | Inference | [FP32](face_detection_and_alignment/tensorflow/mtcc/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf)         | Inference | [FP32](image_recognition/tensorflow/densenet169/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception ResNet V2](https://arxiv.org/pdf/1602.07261.pdf) | Inference | [Int8](image_recognition/tensorflow/inception_resnet_v2/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inception_resnet_v2/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv3/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv3/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V4](https://arxiv.org/pdf/1602.07261.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv4/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv4/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf)        | Inference | [Int8](image_recognition/tensorflow/mobilenet_v1/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/mobilenet_v1/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)          | Inference | [Int8](image_recognition/tensorflow/resnet101/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet101/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)           | Inference | [Int8](image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 50v1.5*](https://github.com/tensorflow/models/tree/master/official/resnet) | Inference | [Int8](image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions) |
| Image Segmentation     | TensorFlow    | [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)          | Inference | [FP32](image_segmentation/tensorflow/maskrcnn/README.md#fp32-inference-instructions) |
| Language Modeling      | TensorFlow    | [LM-1B](https://arxiv.org/pdf/1602.02410.pdf)               | Inference | [FP32](language_modeling/tensorflow/lm-1b/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | [GNMT](https://arxiv.org/pdf/1609.08144.pdf)                | Inference | [FP32](language_translation/tensorflow/gnmt/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | [GNMT](https://arxiv.org/pdf/1609.08144.pdf)                | Training  | [FP32](language_translation/tensorflow/gnmt/README.md#fp32-training-instructions) |
| Object Detection       | TensorFlow    | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)               | Inference | [Int8](object_detection/tensorflow/rfcn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/rfcn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)        | Inference | [Int8](object_detection/tensorflow/faster_rcnn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/faster_rcnn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)       | Inference | [Int8](object_detection/tensorflow/ssd-mobilenet/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/ssd-mobilenet/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf)        | Inference | [Int8](object_detection/tensorflow/ssd-resnet34/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/ssd-resnet34/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf)        | Training  | [FP32](object_detection/tensorflow/ssd-resnet34/README.md#fp32-training-instructions) |
| Object Detection       | TensorFlow    | [SSD-VGG16](https://arxiv.org/pdf/1512.02325.pdf)           | Inference | [Int8](object_detection/tensorflow/ssd_vgg16/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/ssd_vgg16/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [NCF](https://arxiv.org/pdf/1708.05031.pdf)                 | Inference | [FP32](recommendation/tensorflow/ncf/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf)	| Inference | [Int8](recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions) [FP32](recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf)	| Training | [FP32](recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-training-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf)         | Inference | [FP32](recommendation/tensorflow/wide_deep/README.md#fp32-inference-instructions) |

*Means the model is belong to [MLPerf](https://mlperf.org/) models, will long term support. 
