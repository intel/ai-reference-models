<!--- 0. Title -->
# ResNet50 v1.5 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 training
using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Datasets

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training_demo.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`training_1_epoch.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`training_full.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |
| [`multi_instance_training_demo.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training_demo.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 1 for 50 steps. |
| [`multi_instance_training.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh) | Uses mpirun to execute 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16 or bfloat32). Checkpoint files and logs for each instance are saved to the output directory.|

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
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.21.0 (only required for multi-instance)
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
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.21.0 (only required for multi-instance)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Set the environment variables, navigate to your model zoo directory and run quickstart scripts. See the [list of quickstart scripts](#quick-start-scripts) for details on the different options.
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the ImageNet TF records>
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
export PRECISION=<set the precision to "fp32" or "bfloat16" or "bfloat32">
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<int8 precision>](<int8 advanced readme link>) [BFloat16](bfloat16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/resnet50v1-5-fp32-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/resnet50v1-5-fp32-training-tensorflow-container.html).
