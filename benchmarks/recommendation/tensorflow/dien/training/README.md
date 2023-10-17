<!--- 0. Title -->
#  DIEN training

<!-- 10. Description -->
## Description

This document has instructions for running DIEN training using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Use [prepare_data.sh](https://github.com/alibaba/ai-matrix/blob/master/macro_benchmark/DIEN_TF2/prepare_data.sh) to get [a subset of the Amazon book reviews data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) and process it.
Or download and extract the preprocessed data files directly:
```
wget https://zenodo.org/record/3463683/files/data.tar.gz
wget https://zenodo.org/record/3463683/files/data1.tar.gz
wget https://zenodo.org/record/3463683/files/data2.tar.gz

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
Set the `DATASET_DIR` to point to the directory with the dataset files when running DIEN.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training.sh`](/quickstart/recommendation/tensorflow/dien/training/cpu/training.sh) | Runs training with a batch size of 128 for the specified precisions fp32, bfloat16 and bfloat32. |

## Run the model
* Follow the instructions to setup your bare metal environment on Linux systems. Ensure that you have a clone of the [AI Reference Models Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

## Run on Linux
```
# cd to your AI Reference Models directory
cd models

# Set env vars
export DATASET_DIR=<path to the DIEN dataset>
export PRECISION=<set precision to fp32 or bfloat16 or bfloat32>
export OUTPUT_DIR=<path to the directory where log files will be written>

# Run a quickstart script
./quickstart/recommendation/tensorflow/dien/training/cpu/training.sh
```

<!--- 90. Resource Links-->
## Additional Resources
* To run more advanced use cases, see the instructions [here](/benchmarks/recommendation/tensorflow/dien/inference/Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [DevCatalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container
