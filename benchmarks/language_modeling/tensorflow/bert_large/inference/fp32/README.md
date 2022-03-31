<!--- 0. Title -->
# BERT Large FP32 inference

<!-- 10. Description -->

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) FP32 inference
using Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

If you run on Windows, please use a browser to download and extract the dataset files.
For Linux, run:
```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_benchmark.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/fp32_benchmark.sh) | This script runs bert large fp32 inference. |
| [`fp32_profile.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/fp32_profile.sh) | This script runs fp32 inference in profile mode. |
| [`fp32_accuracy.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/fp32_accuracy.sh) | This script is runs bert large fp32 inference in accuracy mode. |
| [`multi_instance_batch_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 128. |
| [`multi_instance_online_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores for each instance with a batch size of 1. |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit on Linux</th>
    <th>Setup without AI Kit on Linux</th>
    <th>Setup without AI Kit on Windows</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit on Linux you will need:</p>
      <ul>
        <li>numactl
        <li>unzip
        <li>wget
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>numactl
        <li>unzip
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/tensorflow/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After your setup is done, download and unzip the pretrained model.
If you run on Windows, please use a browser to download and extract the checkpoint files and pretrained model using the links below.
The path to this directory should be set as the `CHECKPOINT_DIR` before running quickstart scripts.
For Linux, run:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

Download the frozen graph. The path to this file should be set in the
`PRETRAINED_MODEL` environment variable before running the model.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/fp32_bert_squad.pb
PRETRAINED_MODEL=$(pwd)/fp32_bert_squad.pb
```

Next, set environment variables with paths to the [dataset](#datasets),
checkpoint files, pretrained model, and an output directory, then run a
quickstart script on either Linux or Windows systems. See the [list of quickstart scripts](#quick-start-scripts)
for details on the different options.

### Run on Linux
The snippet below shows how to run a quickstart script on Linux:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset being used>
export OUTPUT_DIR=<directory where log files will be saved>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export PRETRAINED_MODEL=<path to the frozen graph>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/fp32/<script name>.sh
```

### Run on Windows
The snippet below shows how to run a quickstart script on Windows systems:

> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\wwm_uncased_L-24_H-1024_A-16`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\wwm_uncased_L-24_H-1024_A-16
> /d/user/wwm_uncased_L-24_H-1024_A-16
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/wwm_uncased_L-24_H-1024_A-16`.

```
# cd to your model zoo directory
cd models

set DATASET_DIR=<path to the dataset being used>
set OUTPUT_DIR=<directory where log files will be saved>
set CHECKPOINT_DIR=<path to the pretrained model checkpoints>
set PRETRAINED_MODEL=<path to the frozen graph>

# Run a script for your desired usage (`fp32_benchmark.sh`, `fp32_accuracy.sh`, or `fp32_profile.sh`)
bash quickstart\language_modeling\tensorflow\bert_large\inference\cpu\fp32\<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/bert-large-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/bert-large-fp32-inference-tensorflow-container.html).

