<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>wget
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>git
        <li>numactl
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After you've completed the setup above, download and extract the pretrained
model. Set the directory path to the `PRETRAINED_MODEL` environment variable.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.tar.gz
tar -xzvf wide_deep_fp32_pretrained_model.tar.gz
export PRETRAINED_MODEL=wide_deep_fp32_pretrained_model
```

<model name> <mode> also uses code from the [TensorFlow models repo](https://github.com/tensorflow/models).
Clone the repo using the PR in the snippet below and save the directory path
to the `TF_MODELS_DIR` environment variable.
```
# We going to use a branch based on older version of the tensorflow model repo.
# Since, we need to to use logs utils on that branch, which were removed from
# the latest master
git clone https://github.com/tensorflow/models.git tensorflow-models
pushd tensorflow-models
git fetch origin pull/7461/head:wide-deep-tf2
git checkout wide-deep-tf2
export TF_MODELS_DIR=$(pwd)
popd
```

Once your environment is setup, navigate back to your model zoo directory and set
environment variables pointing to your dataset and an output directory for log files.
Ensure that you also have the pretrained model and TensorFlow models repo paths
set from the previous steps. Select a [quickstart script](#quick-start-scripts)
to run.
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the Wide & Deep dataset directory>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<pretrained model directory>
export TF_MODELS_DIR=<path to tensorflow-models directory>

./quickstart/recommendation/tensorflow/wide_deep/inference/cpu/fp32/<script name>.sh
```
