# Stable Diffusion Inference

Stable Diffusion inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/stabilityai/stable-diffusion-2-1       |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#installation)

## Bare Metal
### General setup

[Follow](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to install build Pytorch, IPEX, TorchVison and TCMalloc.

### Model Specific Setup
* Set Tcmalloc Preload for better performance
The tcmalloc should be built from the [General setup](#general-setup) section.
```bash
    export LD_PRELOAD="path/lib/libtcmalloc.so":$LD_PRELOAD
```

* Set IOMP preload for better performance
IOMP should be installed in your conda env from the [General setup](#general-setup) section.
```bash
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use fp16 AMX if you are using a supported platform
```bash
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

### Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset will be downloaded. This environment variable will be used again when running training scripts.
```
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

# Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/stable_diffusion/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **DISTRIBUTED** (Only for ACCURACY)              | `export DISTRIBUTED=TRUE`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **DATASET_DIR**       |          `export DATASET_DIR=<path_to_dataset_dir>`             |
| **MODE**      | `export MODE=<choose from: eager, ipex-jit, compile-ipex, compile-inductor>`     |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8-fp32, int8-bf16) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **NNODES** (required for DISTRIBUTED)              | ` export NNODES=#your_node_number`                  |
| **HOSTFILE** (required for DISTRIBUTED)              | `export HOSTFILE=#your_ip_list_file #one ip per line`                  |
| **LOCAL_BATCH_SIZE** (optional for DISTRIBUTED)    |                               `export LOCAL_BATCH_SIZE=64`                                |
7. Run `run_model.sh`

* NOTE:
Please get quantized model before running `INT8-BF16` or `INT8-FP32`.
For `ipex-jit` mode, please [refer](https://github.com/intel/intel-extension-for-transformers/blob/v1.5/examples/huggingface/pytorch/text-to-image/quantization/qat/README.md).
For `compile-inductor` mode, please do calibration first:
  ```
  bash do_calibration.sh
  ```

## Output

Single-tile output will typically looks like:

```
time per prompt(s): 107.73
Latency: 107.65 s
Throughput: 0.00929 samples/sec
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 0.00929
   unit: samples/sec
 - key: latency
   value: 107.73
   unit: s
 - key: accuracy
   value: N/A
   unit: FID
```
