<!--- 0. Title -->
# DLRM bf16 training

<!-- 10. Description -->
## Description

This document has instructions for running DLRM bf16 training using
Intel-optimized PyTorch.

<!--- 20. Datasets -->
## Datasets

Prepare your dataset according to the [instruction described here](/models/recommendation/pytorch/dlrm/training/bf16/README.md#4-prepare-dataset)

Set the `DATA_PATH` to point to "<dir/to/save/dlrm_data>" directory when running DLRM.

<!--- 30. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`train_single_node.sh`](train_single_node.sh) | Run 32K global BS with 4 ranks on 1 node (1 CPX6-4s Node). | 

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 40. Bare Metal -->
## Bare Metal

To run on bare metal first, follow the [instruction described here](/models/recommendation/pytorch/dlrm/training/bf16/README.md#1-install-anaconda-30) until section 4.

After installing the prerequisites, Set environment variables
for the path to your `DATA_PATH`then run a 
[quickstart script](#quick-start-scripts).

```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
quickstart/<script name>.sh
```

<!--- 50. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
DLRM bf16 training. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset.
```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATA_PATH=${DATA_PATH} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATA_PATH}:${DATA_PATH} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  model-zoo:intel-python-dlrm-bf16-training \
  /bin/bash quickstart/recommendation/pytorch/dlrm/training/bf16/train_single_node.sh
```

<!--- 70. License -->
## License

[LICENSE](/LICENSE)

