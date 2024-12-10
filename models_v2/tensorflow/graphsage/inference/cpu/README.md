<!--- 0. Title -->
# GraphSage inference

<!-- 10. Description -->

### Description
This document has instructions for running GraphSage inference for FP32, BFloat16, FP16, Int8 and BFloat32. The model based on this [paper](https://arxiv.org/pdf/1706.02216.pdf). Inference is performed for the task of link prediction.

### Dataset

Download and preprocess the Protein-Protein Interaction dataset using the [instructions here](https://snap.stanford.edu/graphsage/ppi.zip).
```bash
wget https://snap.stanford.edu/graphsage/ppi.zip
unzip ppi.zip
```

Set the `DATASET_DIR` to point to this directory when running GraphSAGE.

```bash
Download Frozen graph:
for fp32, bfloat16, fp16 or bfloat32 precision:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/graphsage_frozen_model.pb

for int8 precision:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/graphsage_int8.pb
```

## Run the model

### Run on Linux

Install the Intel-optimized TensorFlow along with model dependencies under [requirements.txt](/models_v2/tensorflow/graphsage/inference/cpu/requirements.txt)

```bash
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the PPI dataset>
export PRECISION=<set the precision to "fp32" or "bfloat16" or "fp16" or "int8" or "bfloat32">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```

### Inference
1. `inference.sh`
Runs realtime inference using a default `batch_size=1` for the specified precision (fp32, bfloat16, fp16, int8, or bfloat32). To run inference for throughtput, set `BATCH_SIZE` environment variable.
```bash
./models_v2/tensorflow/graphsage/inference/cpu/inference.sh
```

2. `inference_realtime_multi_instance.sh`
Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16, fp16, int8, or bfloat32) with 20 steps. Waits for all instances to complete, then prints a summarized throughput value.
```bash
./models_v2/tensorflow/graphsage/inference/cpu/inference_realtime_multi_instance.sh
```

3. `inference_throughput_multi_instance.sh`
Runs multi instance batch inference using 1 socket per instance for the specified precision (fp32, bfloat16, fp16, int8, or bfloat32) with 20 steps. Waits for all instances to complete, then prints a summarized throughput value.
```bash
./models_v2/tensorflow/graphsage/inference/cpu/inference_throughput_multi_instance.sh
```

### Accuracy
```bash
./models_v2/tensorflow/graphsage/inference/cpu/accuracy.sh
```

### Run with XLA enabled
To run GraphSAGE with XLA enabled, set TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" with the above scripts.
```bash
eg: TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" ./models_v2/tensorflow/graphsage/inference/cpu/inference.sh
```
