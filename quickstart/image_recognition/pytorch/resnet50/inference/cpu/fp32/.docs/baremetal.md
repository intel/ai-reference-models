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

# Download and extract the model package
wget <package url>
tar -xzf <package name>
cd <package dir>
./quickstart/<script name>.sh
```
