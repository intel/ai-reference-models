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
