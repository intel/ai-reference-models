<!--- 0. Title -->
# WaveNet FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running WaveNet FP32 inference using
Intel-optimized TensorFlow.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](/quickstart/text_to_speech/tensorflow/wavenet/inference/cpu/fp32/fp32_inference.sh) | Runs inference with a pretrained model |

<!--- 50. AI Tools -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Tools](/docs/general/tensorflow/AITools.md):

<table>
  <tr>
    <th>Setup using AI Tools</th>
    <th>Setup without AI Tools</th>
  </tr>
  <tr>
    <td>
      <p>AI Tools does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Tools you will need:</p>
      <ul>
        <li>Python 3.6 or 3.7
        <li>git
        <li>numactl
        <li>wget
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>librosa==0.5
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

In addition to the requirements specified above, you will also need a clone
of the [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
repo with pull request #352 for the CPU optimizations.  The path to
the cloned repo needs to be set to the `TF_WAVENET_DIR` environment variable
before running a quickstart script.
```
git clone https://github.com/ibab/tensorflow-wavenet.git
cd tensorflow-wavenet/

git fetch origin pull/352/head:cpu_optimized
git checkout cpu_optimized
export TF_WAVENET_DIR=$(pwd)

cd ..
```

Download and extract the pretrained model checkpoint files.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wavenet_fp32_pretrained_model.tar.gz
tar -xvf wavenet_fp32_pretrained_model.tar.gz
export PRETRAINED_MODEL=$(pwd)/wavenet_checkpoints
```

Navigate to the your AI Reference Models directory, then set an environment variable
for an `OUTPUT_DIR` directory where logs will be written and ensure that you
have the `TF_WAVENET_DIR` and `PRETRAINED_MODEL` variables set. Once this
setup is done, you can run the `fp32_inference.sh` [quickstart script](#quick-start-scripts).
```
# cd to your AI Reference Models directory
cd models

export OUTPUT_DIR=<path to the directory where log files will be written>
export TF_WAVENET_DIR=<tensorflow-wavenet directory>
export PRETRAINED_MODEL=<path to the downloaded and extracted checkpoints>

./quickstart/text_to_speech/tensorflow/wavenet/inference/cpu/fp32/fp32_inference.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/wavenet-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/wavenet-fp32-inference-tensorflow-container.html).

