<!--- 0. Title -->
# BERT Large (Hugging Face) inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large inference (using SQuAD dataset) from Hugging Face using TensorFlow.

Here is the original [model](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering). This model has been slightly modified to print benchmarking information such as latency and throughput.

<!--- 40. Datasets -->
## Dataset

The model script will automatically download and load SQuAD dataset from the [datasets hub](https://huggingface.co/datasets/). No additional setup is needed from the user.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs real-time inference using a default `batch_size=1` (on whole socket) for the specified precision (fp32 fp16 bfloat16 or bfloat32). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance with batch size 1 (for precisions: fp32 fp16 bfloat16 or bfloat32) to compute latency. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket with batch size 128 (for precisions: fp32 fp16 bfloat16 or bfloat32) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures inference accuracy for the specified precision (fp32 fp16 bfloat16 or bfloat32). |

### Pre-Trained Model

Download the pretrained model from the link given below. Please set `DATASET_DIR` to point to the location of the pretrained model files on your local system.

* Fp32 fp16 bfloat16 or bfloat32: <br />
```
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_2/bert_hf_pretrained_model.tar.gz
  tar -xzvf bert_hf_pretrained_model.tar.gz
  export DATASET_DIR=$(pwd)/bert_hf_pretrained_model
```

* For benchmarking, `DATASET_DIR` can be empty (OR) set to the path of pre-trained model mentioned in the above link. If it is empty, the model scripts will automatically download
  the pre-trained model and SQuAD dataset from Hugging Face [models](https://huggingface.co/models) and [datasets](https://huggingface.co/datasets) respectively.
* For accuracy measurement, `DATASET_DIR` **cannot** be empty since the [pre-trained model](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/tree/main) from Hugging Face does not reach the published accuracy results. Hence, it needs to be set to the path of the pre-trained model mentioned in the above link.

### Run the model

#### Install requirements

1) Clone the Intel® AI Reference Models repository:
    ```
    git clone https://github.com/IntelAI/models.git
    ```

2) Install [tf-nightly](https://pypi.org/project/tf-nightly/)

3) Install model-specific requirements:
```
pip install -r models_v2/tensorflow/bert_large_hf/inference/cpu/requirements.txt
```

4) Set the required environment variables:
```
cd models
export PRECISION=<specify the precision to run: fp32 fp16 bfloat16 or bfloat32>
export OUTPUT_DIR=<directory where log files will be written>
export DATASET_NAME="squad"

Optional environment variables:
# For a custom batch size, set env var `BATCH_SIZE`. If not specified, model will run with a default value.
export BATCH_SIZE=<customized batch size value>

# For a custom dataset name, set env var `DATASET_NAME`. If not specified, model will use 'squad' dataset as the default.
export DATASET_NAME=<customized dataset name>
```

#### Accuracy
```
export DATASET_DIR=<path-to-pretrained-model>
./models_v2/tensorflow/bert_large_hf/inference/cpu/accuracy.sh
```

#### Real-time inference (latency) -- runs with batch size of 1
```
`DATASET_DIR` is an optional environment variable. If it is not set, the model scripts will automatically download the pretrained model from huggingface.co/models
  export DATASET_DIR=<path-to-pretrained-model>

./models_v2/tensorflow/bert_large_hf/inference/cpu/inference_realtime_multi_instance.sh
```

#### Throughput -- runs with batch size of 128
```
`DATASET_DIR` is an optional environment variable. If it is not set, the model scripts will automatically download the pretrained model from huggingface.co/models
  export DATASET_DIR=<path-to-pretrained-model>

./models_v2/tensorflow/bert_large_hf/inference/cpu/inference_throughput_multi_instance.sh
```

#### Real-time inference (latency) on whole socket -- runs with a batch size of 1
```
`DATASET_DIR` is an optional environment variable. If it is not set, the model scripts will automatically download the pretrained model from huggingface.co/models
  export DATASET_DIR=<path-to-pretrained-model>

The batch size can be varied using `BATCH_SIZE` env variable. This script differs from `inference_realtime_multi_instance.sh` by running the model on whole socket
instead of `CORES_PER_INSTANCE=4`.

./models_v2/tensorflow/bert_large_hf/inference/cpu/inference.sh
```

### Output
Output from a successful run looks as follows:
```
05/07/2024 10:48:57 - INFO - __main__ - Batch size: 128
05/07/2024 10:48:57 - INFO - __main__ - Total examples: 51200, Warmup examples: 6400
05/07/2024 10:48:57 - INFO - __main__ - Benchmark examples: 44800, Benchmark time: 121.07 secs
05/07/2024 10:48:57 - INFO - __main__ - Throughput (examples/sec): 370.03

```
