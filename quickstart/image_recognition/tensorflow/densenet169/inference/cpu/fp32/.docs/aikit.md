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

After your setup is complete, download the pretrained model. The path to this file
should be set to the `PRETRAINED_MODEL` environment variable before running the quickstart scripts.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/densenet169_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/densenet169_fp32_pretrained_model.pb
```

Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>

# Run a script for your desired usage
./quickstart/image_recognition/tensorflow/densenet169/inference/cpu/fp32/<script name>.sh
```
