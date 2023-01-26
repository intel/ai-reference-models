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
      <p>AI Kit does not currently support wide and deep large ds</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3.7 or newer
        <li><a href="https://pypi.org/project/intel-tensorflow">intel-tensorflow>=2.5.0</a>
        <li>numactl
        <li>git
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After completing the setup, download the pretrained model frozen graph
and save the path to the PRETRAINED_MODEL envionment variable.
```
# FP32 Pretrained Model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/wide_deep_fp32_pretrained_model.pb

# Int8 Pretrained Model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_int8_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/wide_deep_int8_pretrained_model.pb
```

Set the environment variables and run quickstart script. See the list of quickstart scripts for details on the different options.

```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to TF records file>
export PRECISION=<set the precision to "int8" or "fp32">
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run inference with an accuracy check
./quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/accuracy.sh

# Run online inference and set NUM_OMP_THREADS
export NUM_OMP_THREADS=1
./quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/online_inference.sh \
--num-intra-threads 1 --num-inter-threads 1
```
