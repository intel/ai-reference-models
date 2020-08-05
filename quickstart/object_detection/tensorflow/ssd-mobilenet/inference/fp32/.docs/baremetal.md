<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
to run the SSD-Mobilenet scripts must be installed in your environment.

Download and untar the SSD-Mobilenet FP32 inference model package:

```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/ssd-mobilenet-fp32-inference.tar.gz
tar -xvf ssd-mobilenet-fp32-inference.tar.gz
```

To run inference with performance metrics:

```
DATASET_LOCATION=<path to the coco tf record file>
OUTPUT_DIR=<directory where log files will be written>

quickstart/fp32_inference.sh
```

To get accuracy metrics:
```
DATASET_LOCATION=<path to the TF record file>
OUTPUT_DIR=<directory where log files will be written>

quickstart/fp32_accuracy.sh
```

