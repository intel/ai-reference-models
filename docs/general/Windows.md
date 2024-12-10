# Using Intel® AI Reference Models on Windows Systems

## Prerequisites for running on bare metal

Basic requirements for running all TensorFlow models on Windows include:
 * Install [Python 3.9+ 64-bit release for Windows](https://www.python.org/downloads/windows/), and add it to your system `%PATH%` environment variable.
 * Download and Install [Microsoft Visual C++ 2022 Redistributable](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
 * [MSYS2](https://www.msys2.org). If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your `%PATH%` environment variable.
 Then, using `cmd.exe`, run:
 ```
    pacman -S git patch unzip
 ```

## TensorFlow models
* Install [tensorflow-intel](https://pypi.org/project/tensorflow-intel/)
* Set `MSYS64_BASH=C:\msys64\usr\bin\bash.exe` environment variable to your system. The path may change based on where have you installed MSYS2 on our system.
* Install the common models dependencies:
     * python-tk
     * libsm6
     * libxext6
     * requests

Individual models may have additional dependencies that need to be
installed before running it. Please follow the instructions in each model documentation.

The following list of models are tested on Windows, please check each model instructions from the `Model Documentation` column based on the available precisions.
>Note that on Windows systems all of the system cores will be used.
>For users of Windows desktop/laptops, it is strongly encouraged to instead use the batch file provided [here](/benchmarks/common/windows_intel1dnn_setenv.bat) to open a Windows command prompt pre-configured with optimized settings to achieve high AI workload performance on Intel hardware (e.g. TigerLake & AlderLake) for image recognition models.

| Use Case                | Model              | Mode      | Model Documentation |
| ----------------------- | ------------------ | --------- | --------------------------------- |
| Image Recognition       | [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | Inference | [Int8 FP32](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/README.md) |
| Language Modeling       | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_modeling/tensorflow/bert_large/inference/README.md) |
| Language Translation    | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_translation/tensorflow/bert/inference/README.md) |
| Recommendation          | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | Inference | [FP32](/benchmarks/recommendation/tensorflow/wide_deep/inference/README.md) |


## PyTorch models
Intel® Extension for PyTorch is currently not supported on Windows.

Install [PyTorch](https://pytorch.org/)
```
pip install torch torchvision
```

The following list of models are tested on Windows, please check each model instructions from the `Model Documentation` column based on the available precisions.

| Use Case                | Model              | Mode      | Model Documentation |
| ----------------------- | ------------------ | --------- | ------------------- |
| Image Recognition       | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | Inference | [FP32](/models_v2/pytorch/resnet50/inference/cpu/README.md) [BFloat16](/models_v2/pytorch/resnet50/inference/cpu/README.md) |
| Object Detection        | [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | Inference  | [FP32](/models_v2/pytorch/maskrcnn/inference/cpu/README.md) |
