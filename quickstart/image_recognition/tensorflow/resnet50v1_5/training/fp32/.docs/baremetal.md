<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-training.tar.gz
tar -xvf resnet50v1_5_fp32_training.tar.gz
cd resnet50v1_5_fp32_training

quickstart/<script name>.sh
```

