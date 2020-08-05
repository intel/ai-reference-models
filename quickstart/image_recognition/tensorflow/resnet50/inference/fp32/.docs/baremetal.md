<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz
cd resnet50_fp32_inference

quickstart/<script name>.sh
```

