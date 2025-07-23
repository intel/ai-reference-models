# Stable Diffusion Training

Stable Diffusion training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://huggingface.co/stabilityai/stable-diffusion-2-1       |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#installation)

## Bare Metal
### General setup

[Follow](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to istall and build Pytorch, IPEX, TorchVison and TCMalloc.

### Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:
```bash
export ACCELERATE_USE_IPEX=False
```

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

Download the [cat-images dataset](https://huggingface.co/datasets/diffusers/cat_toy_example).

Export the `DATASET_DIR` environment variable to specify the directory where the dataset will be downloaded. This environment variable will be used again when running training scripts.

# Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/stable_diffusion/training/cpu`
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
| **DISTRIBUTED** (optional)              | `export DISTRIBUTED=TRUE`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, fp16, bf16) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **NNODES** (required for DISTRIBUTED)              | ` export NNODES=#your_node_number`                  |
| **HOSTFILE** (required for DISTRIBUTED)              | `export HOSTFILE=#your_ip_list_file #one ip per line`                  |
7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
time_to_train(s): 43.84
Loss: nan
Throughput: 0.49113
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 0.49113
   unit: samples/sec
 - key: latency
   value: N/A
   unit: s
 - key: accuracy
   value: N/A
   unit: FID
```
