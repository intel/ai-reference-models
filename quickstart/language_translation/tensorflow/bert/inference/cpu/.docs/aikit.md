<!--- 50. AI Kit -->
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


* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.


## Run on Linux
```
# cd to your model zoo directory
cd models

# Set env vars
export DATASET_DIR=<path to the MRPC directory>
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model which is downloaded under MRPC directory>
export MODEL_SOURCE=<path to the bert directory>

# Run a quickstart script
./quickstart/language_translation/tensorflow/bert/inference/cpu/<script name>.sh
```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

# Set environment variables
set DATASET_DIR=<path to the MRPC directory>
set OUTPUT_DIR=<path to the directory where log files will be written>
set PRETRAINED_MODEL=<path to the pretrained model which is downloaded under MRPC directory>
set MODEL_SOURCE=<path to the bert directory>

# Run a quickstart script
bash quickstart\language_translation\tensorflow\bert\inference\cpu\<script name>.sh
```

> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables.
As an example, if the pretrained model path on Windows is `D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb
> /d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb
>```
>Then, set the `PRETRAINED_MODEL` environment variable `set PRETRAINED_MODEL=/d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb`.
