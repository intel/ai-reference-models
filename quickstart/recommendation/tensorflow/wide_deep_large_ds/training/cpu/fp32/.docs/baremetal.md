<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)

Download and untar the model package and then run a
[quickstart script](#quick-start-scripts) with environment variables
that point to the [dataset](#dataset), a checkpoint directory, and an
output directory where log files and the saved model will be written.

```
DATASET_DIR=<path to the dataset directory>
OUTPUT_DIR=<directory where the logs and the saved model will be written>
CHECKPOINT_DIR=<directory where checkpoint files will be read and written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xvf <package name>
cd <package dir>

./quickstart/<script name>.sh
```

The script will write a log file and the saved model to the `OUTPUT_DIR`
and checkpoints will be written to the `CHECKPOINT_DIR`.

