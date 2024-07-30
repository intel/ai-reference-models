# DLRM v2 Inference

DLRM v2 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm        |           -           |         -          |

# Pre-Requisite
* Host has 4 Intel® Data Center GPU Max and two tiles for each.
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
After downloading and uncompressing the [Criteo 1TB Click Logs dataset](consisting of 24 files from day 0 to day 23), process the raw tsv files into the proper format for training by running ./scripts/process_Criteo_1TB_Click_Logs_dataset.sh with necessary command line arguments.

Example usage:

bash ./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
./criteo_1tb/raw_input_dataset_dir \
./criteo_1tb/temp_intermediate_files_dir \
./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
The script requires 700GB of RAM and takes 1-2 days to run. We currently have features in development to reduce the preproccessing time and memory overhead. MD5 checksums of the expected final preprocessed dataset files are in md5sums_preprocessed_criteo_click_logs_dataset.txt.

the final dataset dir will be like below:
dataset_dir
 |_day_0_dense.npy
 |_day_0_labels.npy
 |_day_0_sparse_multi_hot.npz

this folder will be used as the parameter DATASET_DIR later

wget https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download -O weigths.zip
unzip weights.zip
and the folder will be used as the parameter WEIGHT_DIR later


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/torchrec_dlrm/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest GPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation):
  ```
  python -m pip install torch==<torch_version> torchvision==<torchvision_version> intel-extension-for-pytorch==<ipex_version> --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
  ```
6. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True)                                             |
| **PLATFORM**                 | `export PLATFORM=Max` (Max)                                                 |
| **WEIGHT_DIR**               | `export WEIGHT_DIR=`                                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32768`                              |
| **PRECISION** (optional)     |        `export PRECISION=FP16` (FP16 and FP32 are supported for Max)                 |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
8. Run `run_model.sh`

## Output

Multi-tile output will typically looks like:
```
[0] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03749502139621311 s
[6] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03693882624308268 s
[1] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03849502139621311 s
[3] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03693882624308268 s
[7] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03858977953592936 s
[2] 2024-01-10 21:50:10,779 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.03870058589511448 s
[4] 2024-01-10 21:50:10,780 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.022177388932969836 s
[5] 2024-01-10 21:50:10,780 - __main__ - INFO - avg eval time per iter at ITER: 45, 0.037547969818115236 s
[0] AUROC over test set: 0.8147445321083069.
[0] Number of test samples: 3276800
[0] 2024:01:10-21:50:11:(34779) |CCL_INFO| finalize atl-mpi
[0] 2024:01:10-21:50:11:(34779) |CCL_INFO| finalized atl-mpi
[3] 2024:01:10-21:50:11:(34782) |CCL_INFO| finalizing level-zero
[7] 2024:01:10-21:50:11:(34786) |CCL_INFO| finalizing level-zero
[0] 2024:01:10-21:50:11:(34779) |CCL_INFO| finalizing level-zero
[6] 2024:01:10-21:50:11:(34785) |CCL_INFO| finalizing level-zero
[4] 2024:01:10-21:50:11:(34783) |CCL_INFO| finalizing level-zero
[3] 2024:01:10-21:50:11:(34782) |CCL_INFO| finalized level-zero
[5] 2024:01:10-21:50:11:(34784) |CCL_INFO| finalizing level-zero
[7] 2024:01:10-21:50:11:(34786) |CCL_INFO| finalized level-zero
[0] 2024:01:10-21:50:11:(34779) |CCL_INFO| finalized level-zero
[2] 2024:01:10-21:50:11:(34781) |CCL_INFO| finalizing level-zero
[6] 2024:01:10-21:50:11:(34785) |CCL_INFO| finalized level-zero
[4] 2024:01:10-21:50:11:(34783) |CCL_INFO| finalized level-zero
[5] 2024:01:10-21:50:11:(34784) |CCL_INFO| finalized level-zero
[2] 2024:01:10-21:50:11:(34781) |CCL_INFO| finalized level-zero
[1] 2024:01:10-21:50:11:(34780) |CCL_INFO| finalizing level-zero
[1] 2024:01:10-21:50:11:(34780) |CCL_INFO| finalized level-zero
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 1693411.31
   unit: samples/s
 - key: accuracy
   value: 0.815
   unit: AUROC
```
