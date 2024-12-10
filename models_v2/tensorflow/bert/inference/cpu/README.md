<!--- 0. Title -->
# BERT inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
### Dataset
Download Microsoft Research Paraphrase Corpus (MRPC) data in cloned repository and save it inside `data` folder.
You can also use the helper script [download_glue_data.py](https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3) to download the data:

   ```
   # Obtain a copy of download_glue_data.py to the current directory
   wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
   python3 download_glue_data.py --data_dir ./data/ --tasks MRPC
   ```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference_realtime.sh`](/models_v2/tensorflow/bert/inference/cpu/inference_realtime.sh) | Runs online inference (batch_size=1). |
| [`inference_throughput.sh`](/models_v2/tensorflow/bert/inference/cpu/inference_throughput.sh) | Runs batch inference (batch_size=32). |
| [`accuracy.sh`](/models_v2/tensorflow/bert/inference/cpu/accuracy.sh) | Runs accuracy |


## Run the model
* Download the fp32 "BERT-Base, Uncased" pre-trained model and unzip it inside the dataset directory `data/MRPC`.
If you run on Windows, please use a browser to download the pretrained model using the link below. For Linux, run:

   ```
   cd data/MRPC
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   unzip uncased_L-12_H-768_A-12.zip
   cd ../..
   ```

* Clone [google-research/bert](https://github.com/google-research/bert) repository:
   ```
   git clone --single-branch https://github.com/google-research/bert.git && cd bert/
   git checkout 88a817c37f788702a363ff935fd173b6dc6ac0d6
   ```
   Set the `MODEL_SOURCE` environment variable to the location of `bert` directory.

* Small sets like MRPC have a high variance in the Dev set accuracy, even when starting from the same pre-training checkpoint.
  If you re-run multiple times (making sure to point to different `output-dir`), you should see results between 84% and 88%.


* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Ensure that you have a clone of the [AI Reference Models Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.


## Run on Linux
```
# cd to your AI Reference Models directory
cd models

# Set env vars
export DATASET_DIR=<path to the MRPC directory>
export PRECISION=fp32
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model which is downloaded under MRPC directory>
export MODEL_SOURCE=<path to the bert directory>

# Run a quickstart script
./models_v2/tensorflow/bert/inference/cpu/<script name>.sh
```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

# Set environment variables
set DATASET_DIR=<path to the MRPC directory>
set PRECISION=fp32
set OUTPUT_DIR=<path to the directory where log files will be written>
set PRETRAINED_MODEL=<path to the pretrained model which is downloaded under MRPC directory>
set MODEL_SOURCE=<path to the bert directory>

# Run a quickstart script
bash models_v2\tensorflow\bert\inference\cpu\<script name>.sh
```

> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables.
As an example, if the pretrained model path on Windows is `D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb
> /d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb
>```
>Then, set the `PRETRAINED_MODEL` environment variable `set PRETRAINED_MODEL=/d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb`.

## Additional Resources

* To run more advanced use cases, see the instructions [here](fp32/Advanced.md)
  for calling the `launch_benchmark.py` script directly.
