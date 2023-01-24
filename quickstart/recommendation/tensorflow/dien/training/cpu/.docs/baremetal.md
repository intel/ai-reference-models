<!--- 50. AI Kit -->
## Run the model
* Follow the instructions to setup your bare metal environment on Linux systems. Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

## Run on Linux
```
# cd to your model zoo directory
cd models

# Set env vars
export DATASET_DIR=<path to the DIEN dataset>
export PRECISION=<set precision to fp32 or bfloat16 or bfloat32>
export OUTPUT_DIR=<path to the directory where log files will be written>

# Run a quickstart script
./quickstart/recommendation/tensorflow/dien/training/cpu/training.sh
```
