<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* build-essential
* Cython
* contextlib2
* jupyter
* lxml
* matplotlib
* numpy>=1.17.4
* pillow>=9.3.0
* pycocotools

For more information see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the coco tf record file>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/<script name>.sh
```
