<!--- 50. AI Kit -->
## Run the model

From AI Kit, activate the TensorFlow language modeling environment:
```
conda activate tensorflow_language_modeling
```

If you are not using AI Kit you will need:
* Python 3
* [intel-tensorflow==2.4.0](https://pypi.org/project/intel-tensorflow/)
* git
* numactl
* unzip
* wget
* Clone the Model Zoo repo:
  ```
  git clone https://github.com/IntelAI/models.git
  ```

Download and unzip the pretrained model. The path to this directory should
be set as the `CHECKPOINT_DIR` before running quickstart scripts.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

Download the frozen graph. The path to this file should be set in the
`PRETRAINED_MODEL` environment variable before running the model.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_4_0/asymmetric_per_channel_bert_int8.pb
export PRETRAINED_MODEL=$(pwd)/asymmetric_per_channel_bert_int8.pb
```

Next, set environment variables with paths to the [dataset](#datasets),
checkpoint files, pretrained model, and an output directory, then run a
quickstart script. See the [list of quickstart scripts](#quick-start-scripts)
for details on the different options.

The snippet below shows how to run a quickstart script:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the SQuAD dataset>
export OUTPUT_DIR=<directory where log files will be saved>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export PRETRAINED_MODEL=<path to the frozen graph>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/<script name>.sh
```
