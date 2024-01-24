# DLRM v2 Training

DLRM v2 Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm        |           -           |         -          |

# Pre-Requisite
* Host has 4 Intel® Data Center GPU Max and have two Tiles for each
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* Host has installed [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)

# prepare Dataset
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


## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/torchrec_dlrm/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=65536`                                |
| **PRECISION** (optional)     |                               `export PRECISION=BF16` (BF16, FP32 and TF32 are supported for PVC)                                |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
6. Run `run_model.sh`

## Output


Multi-tile output will typically looks like:
```
[7] 2024-01-10 22:22:36,284 - __main__ - INFO - avg training time per iter at ITER: 45, 0.07149526278177896 s
[7] 2024-01-10 22:22:36,284 - __main__ - INFO - Total number of iterations: 50
[2] 2024-01-10 22:22:36,292 - __main__ - INFO - avg training time per iter at ITER: 45, 0.07994737095303006 s
[2] 2024-01-10 22:22:36,292 - __main__ - INFO - Total number of iterations: 50
[0] 2024-01-10 22:22:36,293 - __main__ - INFO - avg training time per iter at ITER: 45, 0.08237933582729763 s
[0] 2024-01-10 22:22:36,294 - __main__ - INFO - Total number of iterations: 50
[1] 2024-01-10 22:22:36,296 - __main__ - INFO - avg training time per iter at ITER: 45, 0.08394240803188747 s
[1] 2024-01-10 22:22:36,296 - __main__ - INFO - Total number of iterations: 50
[6] 2024-01-10 22:22:36,304 - __main__ - INFO - avg training time per iter at ITER: 45, 0.09519488016764323 s
[6] 2024-01-10 22:22:36,304 - __main__ - INFO - Total number of iterations: 50
[5] 2024-01-10 22:22:36,306 - __main__ - INFO - avg training time per iter at ITER: 45, 0.09369233979119194 s
[5] 2024-01-10 22:22:36,306 - __main__ - INFO - Total number of iterations: 50
[3] 2024-01-10 22:22:36,309 - __main__ - INFO - avg training time per iter at ITER: 45, 0.09690533743964301 s
[3] 2024-01-10 22:22:36,309 - __main__ - INFO - Total number of iterations: 50
[4] 2024-01-10 22:22:36,339 - __main__ - INFO - avg training time per iter at ITER: 45, 0.11025158564249675 s
[4] 2024-01-10 22:22:36,339 - __main__ - INFO - Total number of iterations: 50
[0] 2024:01:10-22:22:37:(38583) |CCL_INFO| finalize atl-mpi
[0] 2024:01:10-22:22:37:(38583) |CCL_INFO| finalized atl-mpi
[3] 2024:01:10-22:22:37:(38586) |CCL_INFO| finalizing level-zero
[4] 2024:01:10-22:22:37:(38587) |CCL_INFO| finalizing level-zero
[3] 2024:01:10-22:22:37:(38586) |CCL_INFO| finalized level-zero
[2] 2024:01:10-22:22:37:(38585) |CCL_INFO| finalizing level-zero
[6] 2024:01:10-22:22:37:(38589) |CCL_INFO| finalizing level-zero
[0] 2024:01:10-22:22:37:(38583) |CCL_INFO| finalizing level-zero
[7] 2024:01:10-22:22:37:(38590) |CCL_INFO| finalizing level-zero
[4] 2024:01:10-22:22:37:(38587) |CCL_INFO| finalized level-zero
[5] 2024:01:10-22:22:37:(38588) |CCL_INFO| finalizing level-zero
[2] 2024:01:10-22:22:37:(38585) |CCL_INFO| finalized level-zero
[6] 2024:01:10-22:22:37:(38589) |CCL_INFO| finalized level-zero
[0] 2024:01:10-22:22:37:(38583) |CCL_INFO| finalized level-zero
[1] 2024:01:10-22:22:37:(38584) |CCL_INFO| finalizing level-zero
[7] 2024:01:10-22:22:37:(38590) |CCL_INFO| finalized level-zero
[5] 2024:01:10-22:22:37:(38588) |CCL_INFO| finalized level-zero
[1] 2024:01:10-22:22:37:(38584) |CCL_INFO| finalized level-zero
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 594422.29
   unit: samples/s
 - key: accuracy
   value: None
   unit: AUROC
```
