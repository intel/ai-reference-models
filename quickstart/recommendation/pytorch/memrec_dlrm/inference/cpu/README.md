<!--- 0. Title -->
# PyTorch MEMREC-DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running MEMREC-DLRM inference using
Intel-optimized PyTorch for bare metal.

## Bare Metal
Setting up MEMREC-DLRM:
-----------------
- Create conda environment: ```conda create -n memrec python=3.11```
- Activate conda environment: ```conda activate memrec```
- Install Intel® Extension for PyTorch:
    ```pip install intel-extension-for-pytorch```
- Install torch: 
  - CPU: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```
- Install mlperf logger:
  - ```git clone https://github.com/mlperf/logging.git mlperf-logging```
  - ```pip install -e mlperf-logging```
- Install dependencies: ```pip install -r quickstart/recommendation/pytorch/memrec_dlrm/requirements.txt```
- Install memrec encoder:
  - ```cd models/recommendation/pytorch/memrec_dlrm/memrec_encoder/code/```
  - ```python setup.py install --user```

## Datasets

### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is
used to run MEMREC. To download the dataset, you will need to visit the Criteo website and accept
their terms of use:
[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
Copy the download URL into the command below as the `<download url>` and
replace the `<dir/to/save/MEMREC_data>` to any path where you want to download
and save the dataset.
```bash
export DATASET_DIR=<dir/to/save/MEMREC_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when MEMREC is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32). |

## Run the model

Download the [MEMREC-DLRM PyTorch weights](https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/memrec_CriteoTB_D75000dw75000_K1kw1.pt)
and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the memrec_CriteoTB_D75000dw75000_K1kw1.pt file>
```

Follow the instructions above to setup your bare metal environment, do the
model-specific setup and download and prepropcess the dataset. Once all the
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
export WEIGHT_PATH=<path to the memrec pt file> # only needed for testing accuracy
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run a quickstart script (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/memrec_dlrm/inference/cpu
bash inference_performance.sh

# Evaluate Inference Accuracy (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/memrec_dlrm/inference/cpu
bash accuracy.sh
```

<!--- 80. License -->
## License

[LICENSE](../../../../../../models/recommendation/pytorch/memrec_dlrm/memrec_encoder/LICENSE)

[THIRD PARTY LICENSES](../../../../../../models/recommendation/pytorch/memrec_dlrm/memrec_encoder/THIRD-PARTY-LICENSES)
