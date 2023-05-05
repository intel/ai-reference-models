<!--- 0. Title -->
# MMoE Training

<!-- 10. Description -->

### Description
This document has instructions for training the Multi-gate Mixture of Experts (MMoE) model in FP32 precision. The model is based on this [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007). MMoE is used to model task relationships from the data in order to perform multiple-task learning. Multi-task learning is typically used in recommendation systems.

### Dataset

We use the [UCI census-income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) to train the MMoE model. Download the dataset from the [UCI dataset archive](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/) using the [`download_data.py`](../../../../../models/recommendation/tensorflow/mmoe/download_data.py) script.


For training, set the `DATASET_DIR` to point to the `census-income.data.gz` file containing the training data. _Note that the model script will take care of extracting the compressed data_.

## Run the model

### Run on Linux

Install the Intel-optimized TensorFlow along with model dependencies under [requirements.txt](../../../../../models/recommendation/tensorflow/mmoe/training/requirements.txt)

```
# cd to your model zoo directory
cd models
export DATASET_DIR=<path to the test data>
export PRECISION=<set the precision to "fp32" or "bfloat16" or "fp16">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/recommendation/tensorflow/mmoe/training/cpu/training.sh
```
