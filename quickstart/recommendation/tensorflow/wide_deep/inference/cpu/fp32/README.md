<!--- 0. Title -->
# Wide & Deep FP32 inference

<!-- 10. Description -->

This document has instructions for running Wide & Deep FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[wide-deep-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wide-deep-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Dataset
Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running 
following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/wide_deep/census_dataset.py)
Please note that below program requires `requests` module to be installed. You can install it using `pip install requests`.
Dataset will be downloaded in directory provided using `--data_dir`. If you are behind corporate proxy, then you can provide proxy URLs
using `--http_proxy` and `--https_proxy` arguments.
```
git clone https://github.com/IntelAI/models.git
cd models
python ./benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir /home/<user>/widedeep_dataset
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference_online.sh`](fp32_inference_online.sh) | Runs wide & deep model inference online mode (batch size = 1)|
| [`fp32_inference_batch.sh`](fp32_inference_batch.sh) | Runs wide & deep model inference in batch mode (batch size = 1024)|

<!--- 50. Bare Metal -->
### Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

1. Download and untar the Wide & Deep FP32 inference model package:

    ```
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wide-deep-fp32-inference.tar.gz
    tar -xvf wide-deep-fp32-inference.tar.gz
    ```

2. Clone `tensorflow/models` as a `tensorflow-models`
       
    ```
    # We going to use a branch based on older version of the tensorflow model repo.
    # Since, we need to to use logs utils on that branch, which were removed from 
    # the latest master
    git clone https://github.com/tensorflow/models.git tensorflow-models
    cd tensorflow-models
    git fetch origin pull/7461/head:wide-deep-tf2
    git checkout wide-deep-tf2
    ```

3. Once your environment is setup, navigate back to the directory that contains the Wide & Deep FP32 inference
   model package, set environment variables pointing to your dataset and output directories, and then run
   a quickstart script.
    ```
    DATASET_DIR=<path to the Wide & Deep dataset directory>
    OUTPUT_DIR=<directory where log files will be written>
    TF_MODELS_DIR=<path to tensorflow-models>

    ./quickstart/<script name>.sh
    ```

<!-- 60. Docker -->
### Docker

When running in docker, the Wide & Deep FP32 inference container includes the model package and TensorFlow model source repo,
which is needed to run inference. To run the quickstart scripts, you'll need to provide volume mounts for the dataset and
an output directory where log files will be written.

```
DATASET_DIR=<path to the Wide & Deep dataset directory>
OUTPUT_DIR=<directory where log files will be written>

docker run \
--env DATASET_DIR=${DATASET_DIR} \
--env OUTPUT_DIR=${OUTPUT_DIR} \
--env http_proxy=${http_proxy} \
--env https_proxy=${https_proxy} \
--volume ${DATASET_DIR}:${DATASET_DIR} \
--volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
--privileged --init -t \
intel/recommendation:tf-latest-wide-deep-fp32-inference \
/bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

