<!--- 0. Title -->
# ResNet50 FP32 inference

<!-- 10. Description -->

This document has instructions for running ResNet50 BFloat16 inference using
[intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

Note that the ImageNet dataset is used in these ResNet50 examples (only for accuracy script).
- Download and extract the ImageNet2012 dataset from http://www.image-net.org/
    - and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
<!--- 20. Download link -->
## Download link

[resnet50-bf16-inference-pytorch.tar.gz](https://drive.google.com/file/d/1jQpwfPbwxy_yZ6AFVz24l_SONnawywPN/view?usp=sharing)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bf16_online_inference.sh`](bf16_online_inference.sh) | Runs online inference using synthetic data (batch_size=1). |
| [`bf16_batch_inference.sh`](bf16_batch_inference.sh) | Runs batch inference using synthetic data (batch_size=128). |
| [`bf16_accuracy.sh`](bf16_accuracy.sh) | Measures the model accuracy (batch_size=128). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch)
* [torchvision==v0.6.1](https://github.com/pytorch/vision/tree/v0.6.1)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
# Optional: to run accuracy script
export DATASET_DIR=<path to the preprocessed imagenet dataset>

# Download model package from https://drive.google.com/file/d/1jQpwfPbwxy_yZ6AFVz24l_SONnawywPN/view?                usp=sharing
tar -xzf resnet50-bf16-inference-pytorch.tar.gz
cd resnet50-bf16-inference-pytorch

bash quickstart/<script name>.sh
```


<!-- 60. Docker -->
## Docker
TODO

<!-- 61. Advanced Options -->
<!--- 80. License -->
## License

[LICENSE](/LICENSE)

