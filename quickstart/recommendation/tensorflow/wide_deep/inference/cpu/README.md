<!--- 0. Title -->
# Wide & Deep inference

<!-- 10. Description -->

This document has instructions for running Wide & Deep inference using
Intel-optimized TensorFlow.

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
| [`inference_online.sh`](/quickstart/recommendation/tensorflow/wide_deep/inference/cpu/inference_online.sh) | Runs wide & deep model inference online mode (batch size = 1)|
| [`inference_batch.sh`](/quickstart/recommendation/tensorflow/wide_deep/inference/cpu/inference_batch.sh) | Runs wide & deep model inference in batch mode (batch size = 1024)|

<!-- 60. Docker -->
### Docker

When running in docker, the Wide & Deep <precision> inference container includes the model package and TensorFlow model source repo,
which is needed to run inference. To run the quickstart scripts, you'll need to provide volume mounts for the dataset and
an output directory where log files will be written.

```
DATASET_DIR=<path to the Wide & Deep dataset directory>
PRECISION=fp32
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
--env DATASET_DIR=${DATASET_DIR} \
--env PRECISION=${PRECISION} \
--env OUTPUT_DIR=${OUTPUT_DIR} \
--env BATCH_SIZE=${BATCH_SIZE} \
--env http_proxy=${http_proxy} \
--env https_proxy=${https_proxy} \
--volume ${DATASET_DIR}:${DATASET_DIR} \
--volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
--privileged --init -t \
intel/recommendation:tf-latest-wide-deep-inference \
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

