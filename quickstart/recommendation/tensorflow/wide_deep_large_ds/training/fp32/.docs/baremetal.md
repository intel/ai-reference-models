<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow](https://pypi.org/project/intel-tensorflow/)

Download and untar the model package and then run a
[quickstart script](#quick-start-scripts) with enviornment variables
that point to the [dataset](#dataset) and an output directory where
log files, checkpoint files, and the saved model will be written.

```
DATASET_DIR=<path to the dataset directory>
OUTPUT_DIR=<directory where the logs, checkpoints, and the saved model will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/cicd-or-local/model-zoo/wide-deep-large-ds-fp32-training.tar.gz
tar -xvf wide-deep-large-ds-fp32-training.tar.gz
cd wide-deep-large-ds-fp32-training.tar.gz

quickstart/<script name>.sh
```

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

