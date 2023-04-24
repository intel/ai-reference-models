<!--- 0. Title -->
# MMoE Inference

<!-- 10. Description -->

### Description
This document has instructions for running inference using the Multi-gate Mixture of Experts (MMoE) model in FP32 precision. The model is based on this [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007). MMoE is used to model task relationships from the data in order to perform multiple-task learning. Multi-task learning is typically used in recommendation systems.

### Dataset

We use the [UCI census-income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) to perform inference using the MMoE model. Download the dataset from the [UCI dataset archive](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/) using the [`download_data.py`](../../../../../models/recommendation/tensorflow/mmoe/download_data.py) script.


For inference, set the `DATASET_DIR` to point to the `census-income.test.gz` file containing the test data.

```
Download Frozen graph:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_11_0/mmoe_pretrained_frozen_fp32.pb
```

## Run the model

### Run on Linux
```
# cd to your model zoo directory
cd models
export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the test data>
export PRECISION=<set the precision to "fp32">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```

* Inference
```
./quickstart/recommendation/tensorflow/mmoe/inference/cpu/inference.sh
```

* Accuracy
```
./quickstart/recommendation/tensorflow/mmoe/inference/cpu/accuracy.sh
```
