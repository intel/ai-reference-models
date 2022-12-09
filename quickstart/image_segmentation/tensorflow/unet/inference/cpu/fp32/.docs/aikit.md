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
      <p>AI Kit does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>numactl
        <li>numpy==1.16.3
        <li>Pillow>=9.3.0
        <li>matplotlib
        <li>click
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>


Running <model name> also requires a clone of the
[tf_unet](https://github.com/jakeret/tf_unet) repository with [PR #276](https://github.com/jakeret/tf_unet/pull/276)
to get cpu optimizations. Set the `TF_UNET_DIR` env var to the path of your clone.
```
git clone https://github.com/jakeret/tf_unet.git
cd tf_unet/
git fetch origin pull/276/head:cpu_optimized
git checkout cpu_optimized
export TF_UNET_DIR=$(pwd)
cd ..
``` 

Download and extract the pretrained model and set the path to the
`PRETRAINED_MODEL` env var.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/unet_fp32_pretrained_model.tar.gz
tar -xvf unet_fp32_pretrained_model.tar.gz
export PRETRAINED_MODEL=$(pwd)/unet_trained
```

After your environment is setup, set an environment variable to 
an `OUTPUT_DIR` where log files will be written. Ensure that you already have
the `TF_UNET_DIR` and `PRETRAINED_MODEL` paths set from the previous commands.
Once the environment variables are all set, you can run a
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export OUTPUT_DIR=<path to the directory where log files will be written>
export TF_UNET_DIR=<path to the TF UNet directory tf_unet>
export PRETRAINED_MODEL=<path to the pretrained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_segmentation/tensorflow/unet/inference/cpu/fp32/fp32_inference.sh
```
