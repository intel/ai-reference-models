<!--- 0. Title -->
# Wide & Deep FP32 inference

<!-- 10. Description -->

This document has instructions for running Wide & Deep FP32 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Dataset
Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running 
following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/wide_deep/census_dataset.py)
Please note that below program requires `requests` module to be installed. You can install it using `pip install requests`.
Dataset will be downloaded in directory provided using `--data_dir`. If you are behind corporate proxy, then you can provide proxy URLs
using `--http_proxy` and `--https_proxy` arguments.
```
git clone https://github.com/IntelAI/models.git
cd models
python ./benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir /home/<user>/widedeep_dataset
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference_online.sh`](fp32_inference_online.sh) | Runs wide & deep model inference online mode (batch size = 1)|
| [`fp32_inference_batch.sh`](fp32_inference_batch.sh) | Runs wide & deep model inference in batch mode (batch size = 1024)|

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
        <li>git
        <li>numactl
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
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/tensorflow/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
        <li>git
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After you've completed the setup above, download and extract the pretrained
model. If you run on Windows, please use a browser to download the pretrained model using the link below.
Set the directory path to the `PRETRAINED_MODEL` environment variable.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.tar.gz
tar -xzvf wide_deep_fp32_pretrained_model.tar.gz
export PRETRAINED_MODEL=wide_deep_fp32_pretrained_model
```

Wide & Deep inference also uses code from the [TensorFlow models repo](https://github.com/tensorflow/models).
Clone the repo using the PR in the snippet below and save the directory path
to the `TF_MODELS_DIR` environment variable.
```
# We going to use a branch based on older version of the tensorflow model repo.
# Since, we need to to use logs utils on that branch, which were removed from
# the latest master
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git fetch origin pull/7461/head:wide-deep-tf2
git checkout wide-deep-tf2
```

Once your environment is setup, navigate back to your model zoo directory and set
environment variables pointing to your dataset and an output directory for log files.
Ensure that you also have the pretrained model and TensorFlow models repo paths
set from the previous steps. Select a [quickstart script](#quick-start-scripts)
to run.
### Run on Linux
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the Wide & Deep dataset directory>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<pretrained model directory>
export TF_MODELS_DIR=<path to tensorflow-models directory>

./quickstart/recommendation/tensorflow/wide_deep/inference/cpu/fp32/<script name>.sh
```
### Run on Windows
Using cmd.exe, run:
```
# cd to your model zoo directory
cd models

set PRETRAINED_MODEL=<pretrained model directory>
set DATASET_DIR=<path to the Wide & Deep dataset directory>
set OUTPUT_DIR=<directory where log files will be written>
set TF_MODELS_DIR=<path to tensorflow-models directory>

bash quickstart\recommendation\tensorflow\wide_deep\inference\cpu\fp32\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\<user>\widedeep_dataset`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\<user>\widedeep_dataset
> /d/<user>/widedeep_dataset
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/<user>/widedeep_dataset`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-fp32-inference-tensorflow-container.html).

