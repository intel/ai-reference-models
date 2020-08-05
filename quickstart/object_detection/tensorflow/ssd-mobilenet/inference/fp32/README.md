<!--- 0. Title -->
# SSD-Mobilenet FP32 inference

<!-- 10. Description -->

This document has instructions for running SSD-Mobilenet FP32 inference using
Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[ssd-mobilenet-fp32-inference.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/ssd-mobilenet-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Dataset

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The inference and accuracy quickstart scripts require the dataset to be converted into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](fp32_inference.sh) | Runs inference on TF records and outputs performance metrics. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)


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


!<--- 60. Docker -->
## Docker

When running in docker, the SSD-Mobilenet FP32 inference container includes the
libraries and the model package, which are needed to run SSD-Mobilenet FP32
inference. To run the quickstart scripts, you'll need to provide volume mounts for the
[COCO validation dataset](/datasets/coco/README.md) TF Record file and an output directory
where log files will be written.

To run inference with performance metrics:

    ```
    DATASET_DIR=<path to the coco TF Record file>
    OUTPUT_DIR=<directory where log files will be written>

    docker run \
    --env DATASET_LOCATION=${DATASET_LOCATION} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --volume ${DATASET_LOCATION}:${DATASET_LOCATION} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --privileged --init -t \
    amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-object-detection-ssd-mobilenet-fp32-inference \
    /bin/bash quickstart/fp32_inference.sh
    ```

Below is a sample log file tail when running for inference:

    ```
    Batchsize: 1
    Time spent per BATCH:    46.7484 ms
    Total samples/sec:    21.3911 samples/s
    Log file location: ${OUTPUT_DIR}/benchmark_ssd-mobilenet_inference_fp32_20200731_203206.log
    ```

To get accuracy metrics:
    
    ```
    DATASET_LOCATION=<path to the coco TF record file>
    OUTPUT_DIR=<directory where log files will be written>

    docker run \
    --env DATASET_LOCATION=${DATASET_LOCATION} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --volume ${DATASET_LOCATION}:${DATASET_LOCATION} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --privileged --init -t \
    amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-object-detection-ssd-mobilenet-fp32-inference \
    /bin/bash quickstart/fp32_accuracy.sh
    ```

Below is a sample log file tail when running for accuracy:

    ```
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=39.59s).
    Accumulating evaluation results...
    DONE (t=4.17s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.417
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.277
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.262
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
    Log file location: ${OUTPUT_DIR}/benchmark_ssd-mobilenet_inference_fp32_20200731_203756.log
    ```


<!--- 80. License -->
## License

[LICENSE](/LICENSE)


