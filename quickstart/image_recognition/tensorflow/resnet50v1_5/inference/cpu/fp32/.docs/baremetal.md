<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package. The model package includes a
[pretrained model](https://zenodo.org/record/2535873/files/resnet50_v1.pb)
and the scripts needed to run the <model name> <precision> <model>. Set
environment variables to point to the imagenet dataset directory (if real
data is being used), and an output directory where log files will be
written, then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/<script name>.sh
```

