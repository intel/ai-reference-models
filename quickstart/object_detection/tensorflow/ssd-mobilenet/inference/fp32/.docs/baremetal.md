<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
to run the <model name> scripts must be installed in your environment.

To run <mode>, set environment variables with the path to the dataset
and an output directory, download and untar the <model name> <precision>
<mode> model package, and then run a [quickstart script](#quick-start-scripts).
```
DATASET_DIR=<path to the coco tf record file>
OUTPUT_DIR=<directory where log files will be written>

wget <package url>
tar -xzf <package name>
cd <package dir>

quickstart/<script name>.sh
```
