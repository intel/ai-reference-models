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
        <li>numactl
        <li>wget
        <li>Activate the tensorflow conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>[intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
        <li>git
        <li>numactl
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After finishing the setup above, download the pretrained model and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_v1_5_bfloat16.pb
export PRETRAINED_MODEL=$(pwd)/resnet50_v1_5_bfloat16.pb
```

Set environment variables for the path to your `DATASET_DIR` for ImageNet and an 
`OUTPUT_DIR` where log files will be written.  Navigate to your model zoo directory 
and then run a [quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the ImageNet TF records>
export OUTPUT_DIR=<directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/bfloat16/<script name>.sh
```
