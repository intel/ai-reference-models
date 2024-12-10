<!--- 0. Title -->
# TensorFlow BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large inference on baremetal using
Intel-optimized TensorFlow.

<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create a virtual environment `venv-tf`:
```
python -m venv venv-tf
source venv-tf/bin/activate
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow
```

* Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment:
  ```bash
  DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  # To run VNNI, please set
  DNNL_MAX_CPU_ISA=AVX512_CORE_BF16
  ```

* Clone [Intel AI Reference Models repository](https://github.com/IntelAI/models)
  ```bash
  git clone https://github.com/IntelAI/models
  ```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weight_sharing.sh` | Runs multi instance realtime inference with weight sharing for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency for weight sharing. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with batch size 128 (for precisions: fp32, int8 or bfloat16) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32, int8 or bfloat16 and bfloat32). |

<!--- 30. Datasets -->
## Datasets

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```bash
# INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/per_channel_opt_int8_bf16_bert.pb

#FP32 and BFloat32:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/fp32_bert_squad.pb

#BFloat16:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/optimized_bf16_bert.pb
```

## Download checkpoints:
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

## Run the model

Set environment variables to specify the dataset directory, precision to run, path to pretrained files and an output directory.
```
# Navigate to the models directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32 , bfloat32 and bfloat16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>

#Optional envs
export BATCH_SIZE=<customized batch size value, otherwise it will run with the default value>
export OMP_NUM_THREADS=<customized value for omp_num_threads, otherwise it will run with the default value>
export CORES_PER_INSTANCE=<customized value for cores_per_instance, otherwise it will run with the default value>

Run the script:
./models_v2/tensorflow/bert_large/inference/cpu/<script_name.sh>
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

