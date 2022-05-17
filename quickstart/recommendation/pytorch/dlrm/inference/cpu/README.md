<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running DLRM inference using
Intel-optimized PyTorch for bare metal.

## Bare Metal
### General Setup
Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, and Jemalloc.

### Model Specific Setup

* Install dependencies
  ```bash
  cd <clone of the model zoo>/quickstart/recommendation/pytorch/dlrm
  pip install requirements.txt
  ```

* Set Jemalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```bash
  export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env from the [General setup](#general-setup) section.
  ```bash
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```bash
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Datasets

### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is
used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept
their terms of use:
[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
Copy the download URL into the command below as the `<download url>` and
replace the `<dir/to/save/dlrm_data>` to any path where you want to download
and save the dataset.
```bash
export DATASET_DIR=<dir/to/save/dlrm_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |

## Run the model

Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the
[MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights)
and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the tb00_40M.pt file>
```

Follow the instructions above to setup your bare metal environment, do the
model-specific setup and download and prepropcess the datsaet. Once all the
setup is done, the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory,
precision, weights file, and an output directory.

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export PRECISION=<specify the precision to run>
export WEIGHT_PATH=<path to the tb00_40M.pt file> # only needed for testing accuracy
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run a quickstart script (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/inference/cpu
bash inference_performance.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
