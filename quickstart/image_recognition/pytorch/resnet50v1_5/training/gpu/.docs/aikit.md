<!--- 50. AI Kit -->
## Run the model

Requirements:
* Host machine has Intel GPU.
* Host machine has installed Linux kernel that is compatible with GPU drivers.
* `lspci` (installed using `pciutils` from apt or yum)
* Source the oneAPI AI Kit `setvars.sh` file (once per session)
   ```
   source /opt/intel/oneapi/setvars.sh
   ```
* Setup a conda environment with the dependencies needed to run SSD-ResNet34. Clone
  the pytorch conda environment from AI Kit before running the setup script.
  ```
  # Create a clone of the AI Kit pytorch conda environment before running the setup.py script
  conda create --clone pytorch --name ${USER}-pytorch-resnet50
  conda activate ${USER}-pytorch-resnet50

  # Navigate to the <model name> directory, and the setup.sh script from the quickstart folder
  cd <package dir>
  quickstart/setup.sh
  ```
  Note that the same conda environment can be used for both training and inference.

See the [datasets section](#datasets) of this document for instructions on
downloading and extracting the ImageNet dataset.

This snippet shows how to run a quickstart script using AI Kit. Before running
<mode>, you'll need to make sure that you have all the requirements listed above,
including the conda environment activated. Set environment variables for the path to
your dataset, an output directory, and specify the precision to run.
```
# Navigate to the <model name> <mode> directory
cd <package dir>

# Activate the PyTorch <model name> conda environment
conda activate ${USER}-pytorch-resnet50

# Set environment vars for the dataset and an output directory
export DATASET_DIR=<path the ImageNet directory>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run (fp32 or bf16)>

# Run a quickstart script
quickstart/<script name>.sh
```
