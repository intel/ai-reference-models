<!--- 0. Title -->
# TensorFlow GPT_J_6GB Fine tuning

<!-- 10. Description -->

This document has instructions for running GPT_J_6B finetuning
using Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

### GLUE and Lambada Datasets
  Datasets is downloaded by the model automatically. The user need to set cache dir
  where the data will be stored
  Huggingface GLUE dataset : https://huggingface.co/datasets/glue
  Huggingface Lambada dataset : https://huggingface.co/datasets/lambada

```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name      | Description |
|------------------|-------------|
| `run_glue.sh'    | Uses mpirun to execute 1 process per socket for GPT_J_6GB GLUE finetuning with the specified precision (fp32 or bfloat16, or fp16). Logs for each instance are saved to the output directory. |
| `run_lambada.sh' | Uses mpirun to execute 1 process per socket for GPT_J_6GB Lambada finetuning with the specified precision (fp32 or bfloat16, or fp16). Logs for each instance are saved to the output directory. |

<!--- 50. Baremetal -->
## Run the model

### Setup on baremetal

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
        <li>unzip
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.25.0 (only required for multi-instance)
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
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.25.0 (only required for multi-instance)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

### Run on Linux

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
The Huggingface API downloads datasets to the DATASET directory specified.
Make sure this directory has write permissions and has enough space.
This directory will also be used as cache by tokenizer.
Dataset sizes :
    GLUE   : 1GB include 9 GLUE tasks
    Lambada: 335MB

```
# Navigate to the container package directory
cd models

# Set the required environment vars
export DATASET_DIR=<directory where dataset will be downloaded to by Huggingface dataset API>
export PRECISION=<fp32,bfloat16,fp16>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with run_gluesh quickstart script
./quickstart/language_modeling/tensorflow/gpt_j_6B/training/cpu/run_glue.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<fp16 precision>](<fp6/Advanced.md>) [BFloat16](bfloat16/Advanced.md,) for calling the `launch_benchmark.py` script directly.
