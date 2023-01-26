<!--- 0. Title -->
# Transformer LT Official inference

<!-- 10. Description -->

This document has instructions for running
[Transformer Language Translation](https://arxiv.org/pdf/1706.03762.pdf)
 inference using Intel-optimized TensorFlow. The original code for
the Transformer LT Official model came from the
[TensorFlow Model Garden repository](https://github.com/tensorflow/models/tree/v2.2.0/official/nlp/transformer).


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
| [`online_inference.sh`](/quickstart/language_translation/tensorflow/transformer_lt_official/inference/cpu/online_inference.sh) | Runs online inference (batch_size=1). |
| [`batch_inference.sh`](/quickstart/language_translation/tensorflow/transformer_lt_official/inference/cpu/batch_inference.sh) | Runs batch inference (batch_size=64). |

<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts,
libraries, and pretrained model needed to run Transformer LT Official inference. To run one of the model quickstart scripts using this
container, you'll need to provide volume mounts for the dataset and an
output directory.

```
DATASET_DIR=<path to the test dataset directory>
PRECISION=fp32
OUTPUT_DIR=<directory where the log and translation file will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=#{BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/language-translation:tf-latest-transformer-lt-official-inference \
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

