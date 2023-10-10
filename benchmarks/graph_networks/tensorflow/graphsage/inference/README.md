<!--- 0. Title -->
# GraphSage inference

<!-- 10. Description -->

### Description
This document has instructions for running GraphSage inference for FP32, BFloat16 and FP16. The model based on this [paper](https://arxiv.org/pdf/1706.02216.pdf). Inference is performed for the task of link prediction. 


### Dataset

Download and preprocess the Protein-Protein Interaction dataset using the [instructions here](https://snap.stanford.edu/graphsage/ppi.zip).
```
wget https://snap.stanford.edu/graphsage/ppi.zip
unzip ppi.zip
```

Set the `DATASET_DIR` to point to this directory when running GraphSAGE.

```
Download Frozen graph:
for fp32, bfloat16 or fp16 precision:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/graphsage_frozen_model.pb

for int8 precision:
https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/graphsage_frozen_model_int8.pb

```

## Run the model

### Run on Linux

Install the Intel-optimized TensorFlow along with model dependencies under [requirements.txt](../../../../../models/graph_networks/tensorflow/graphsage/inference/requirements.txt)

```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the PPI dataset>
export PRECISION=<set the precision to "fp32" or "bfloat16" or "fp16" or "int8">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```

### Inference
1. `inference.sh`
Runs realtime inference using a default `batch_size=1` for the specified precision (fp32, bfloat16,fp16, or int8). To run inference for throughtput, set `BATCH_SIZE` environment variable.
```
./quickstart/graph_networks/tensorflow/graphsage/inference/cpu/inference.sh
```

2. `inference_realtime_multi_instance.sh`
Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16,fp16, or int8) with 20 steps. Waits for all instances to complete, then prints a summarized throughput value.
```
./quickstart/graph_networks/tensorflow/graphsage/inference/cpu/inference_realtime_multi_instance.sh
```

3. `inference_throughput_multi_instance.sh`
Runs multi instance batch inference using 1 socket per instance for the specified precision (fp32, bfloat16,fp16, or int8) with 20 steps. Waits for all instances to complete, then prints a summarized throughput value.
```
./quickstart/graph_networks/tensorflow/graphsage/inference/cpu/inference_throughput_multi_instance.sh
```

### Accuracy
```
./quickstart/graph_networks/tensorflow/graphsage/inference/cpu/accuracy.sh
```
