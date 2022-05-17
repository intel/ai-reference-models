<!--- 0. Title -->
# Transformer LT Official FP32 inference

<!-- 10. Description -->

This document has instructions for running
[Transformer Language Translation](https://arxiv.org/pdf/1706.03762.pdf)
FP32 inference using Intel-optimized TensorFlow. The original code for
the Transformer LT Official model came from the
[TensorFlow Model Garden repository](https://github.com/tensorflow/models/tree/v2.2.0/official/nlp/transformer).


<!--- 20. Download link -->
## Download link

[transformer-lt-official-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/transformer-lt-official-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Dataset

Download and preprocess the English-German dataset using the
[data_download.py](https://github.com/tensorflow/models/blob/v2.2.0/official/nlp/transformer/data_download.py)
script from the [TensorFlow Model Garden repo](https://github.com/tensorflow/models).
The Transformer model README has a section with
[instructions for using the script](https://github.com/tensorflow/models/tree/v2.2.0/official/nlp/transformer#download-and-preprocess-datasets).

Once the script completes, you should have a dataset directory with
the following files: `newstest2014.de`, `newstest2014.en`, and
a vocab text file. For simplicity, rename the vocab file to `vocab.txt`.
The path to the directory with these files should be set as the
`DATASET_DIR` environment variable when using the
[quickstart scripts](#quick-start-scripts).


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/quickstart/language_translation/tensorflow/transformer_lt_official/inference/cpu/fp32/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/quickstart/language_translation/tensorflow/transformer_lt_official/inference/cpu/fp32/fp32_batch_inference.sh) | Runs batch inference (batch_size=64). |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* [Cython](https://pypi.org/project/Cython/)
* [pandas](https://pypi.org/project/pandas/)

Download and untar the model package and then run a
[quickstart script](#quick-start-scripts) with environment variables
that point to your dataset and an output directory.

```
DATASET_DIR=<path to the test dataset directory>
OUTPUT_DIR=<directory where the log and translation file will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/transformer-lt-official-fp32-inference.tar.gz
tar -xzf transformer-lt-official-fp32-inference.tar.gz
cd transformer-lt-official-fp32-inference

./quickstart/<script name>.sh
```

If you have your own pretrained model, you can specify the path to the frozen
graph .pb file using the `PRETRAINED_MODEL` environment variable.


<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts,
libraries, and pretrained model needed to run Transformer LT Official FP32
inference. To run one of the model quickstart scripts using this
container, you'll need to provide volume mounts for the dataset and an
output directory.

```
DATASET_DIR=<path to the test dataset directory>
OUTPUT_DIR=<directory where the log and translation file will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/language-translation:tf-latest-transformer-lt-official-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you have your own pretrained model, you can specify the path to the frozen 
graph .pb file using the `FROZEN_GRAPH` environment variable and mount the
frozen graph's directory as a volume in the container.

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

