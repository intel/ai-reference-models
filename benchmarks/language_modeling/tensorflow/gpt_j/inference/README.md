<!--- 0. Title -->
# TensorFlow GPT-J-6B inference

<!-- 10. Description -->
## Description

This document has instructions for running GPT-J inference using Tensorflow.

<!--- 30. Datasets -->
## Datasets

### GPT-J Data and Model

We use the EleutherAI's LAMBADA dataset to perform inference using the GPT-J model. The inference scripts automatically download and process the data and model for use.

For inferece, set the `CHECKPOINT_DIR` to point to the directory where the model and data exist or where they need to be downloaded and stored.

By default, the script uses an input token size of 1000 and an output token size of 128. These values can be changed by setting the `INPUT_TOKENS` and `MAX_OUTPUT_TOKENS` flags respectively. 

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a custom `batch_size` for the specified precision (fp32, bfloat16 or fp16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference for GPT-J using 14 cores per instance with batch size 1 ( for precisions: fp32, bfloat16 and fp16) to compute latency. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference for GPT-J using 1 instance per socket with batch size 100 (for precisions: fp32, bfloat16 and fp16) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures GPT-J inference perplexity for the specified precision (fp32, bfloat16 and fp16). |

## Run the model

The quickstart scripts can be used as follows

```
# cd to your model zoo directory
cd models

export OUTPUT_DIR=<path to the output directory>
export CHECKPOINT_DIR=<path to the saved model>
export PRECISION=<set the precision to "fp32", "bfloat16" or "fp16">
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
# For a custom number of input tokens, set env var `INPUT_TOKENS` or it will run with a default value of 1000.
export INPUT_TOKENS=<customized input token value>
# For a custom number of max output tokens, set env var `MAX_OUTPUT_TOKENS` or it will run with a default value of 128.
export MAX_OUTPUT_TOKENS=<customized max output token value>

./quickstart/language_modeling/tensorflow/gpt_j/inference/cpu/<script name>.sh
```

## License

Licenses can be found in the model package, in the `licenses` directory.

