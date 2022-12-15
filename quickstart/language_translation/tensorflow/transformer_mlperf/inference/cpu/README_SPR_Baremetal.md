<!--- 0. Title -->
# TensorFlow Transformer Language inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language inference on baremetal using
Intel-optimized TensorFlow.

<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install Intel Tensorflow
  ```bash
  pip install intel-tensorflow==2.11.dev202242
  ```

* Install the keras version that works with the above tensorflow version:
  ```bash
  pip install keras-nightly==2.11.0.dev2022092907
  ```

* Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: 
  ```bash
  DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  # To run VNNI, please set 
  DNNL_MAX_CPU_ISA=AVX512_CORE_BF16
  ```

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models)
  ```bash
  git clone https://github.com/IntelAI/models
  ```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) to compute latency using 4 cores per instance for the specified precision (int8, fp32, bfloat32 or bfloat16). |
| `inference_throughput.sh` | Runs multi instance batch inference with batch-size=448 for precisions (int8, fp32, bfloat16 and bfloat32) to compute throughput using 1 instance per socket. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (int8, fp32, bfloat32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.

Paths to the pb files relative to tf_dataset:
* INT8:
  BERT-Large-squad_int8 = /tf_dataset/pre-trained-models/bert_squad/int8/per_channel_opt_int8_bf16_bert.pb

* FP32 and BFloat32:
  BERT-Large-squad_fp32 = /tf_dataset/pre-trained-models/bert_squad/fp32/new_fp32_bert_squad.pb

* BFloat16:
  BERT-Large-squad_bfloat16 = /tf_dataset/pre-trained-models/bert_squad/bfloat16/optimized_bf16_bert.pb

## Run the model

Set environment variables to
specify the dataset directory, pretrained model path, precision to run, and
an output directory. 
```
# Navigate to the container package directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32, bfloat16 or bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the quickstart scripts:
./quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/<script_name>.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

