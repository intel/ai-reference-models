# PyTorch Latent Consistency Models inference

## Description
This document has instructions for running [Latent Consistency Models (LCMs).](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) inference using Intel-optimized PyTorch.

## Model Information

| Use Case    | Framework   | Model Repository| Branch/Commit| Patch |
|-------------|-------------|-----------------|--------------|--------------|
| Inference   | Pytorch     | https://github.com/huggingface/diffusers.git | v0.23.1 | [`diffusers.patch`](/models_v2/pytorch/LCM/inference/cpu/diffusers.patch) |


## Pre-Requisite

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install and build Pytorch, IPEX, TorchVison, Jemalloc and TCMalloc.

* Install Intel OpenMP
  ```
  pip install packaging intel-openmp accelerate
  ```
* Set IOMP, jemalloc and tcmalloc Preload for better performance
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
  ```
* For distributed accuracy, you will need to install mpirun.

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/LCM/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install general model requirements
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)       | `export TEST_MODE=THROUGHPUT (THROUGHPUT, ACCURACY, REALTIME)`                                  |
| **RUN_MODE**   |   `export RUN_MODE=ipex-jit (specify mode to run: eager, ipex-jit, compile-ipex, compile-inductor)` |
| **DATASET_DIR**              |                        `export DATASET_DIR=<path to the dataset>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <specify the precision to run: fp32, bf32, fp16, bf16, int8-bf16, int8-fp32>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to the directory where log files will be written>`                               |
| **MODEL_DIR** | `export MODEL_DIR=$PWD (set the current path)` |
| **DISTRIBUTED**(Only for Accuracy) | `export DISTRIBUTED=false (Set this to 'true' to run distributed accuracy)`    |
| **BATCH_SIZE** (optional)  |                        `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |
| **TORCH_INDUCTOR** (optional)    | `export TORCH_INDUCTOR=< 0 or 1> (Compile model with PyTorch Inductor backend)`   |

* NOTE:
For `compile-inductor` mode, please do calibration to get quantized model before running `INT8-BF16` or `INT8-FP32`.
  ```
  bash do_calibration.sh
  ```

8. Run `run_model.sh`

## Output
Output typically looks like this:
Running benchmark ...
100%|██████████| 4/4 [00:17<00:00,  4.29s/it]
time per prompt(s): 26.34
100%|██████████| 4/4 [00:17<00:00,  4.29s/it]
time per prompt(s): 26.18
Latency: 26.18 s
Throughput: 0.03820 samples/sec

Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 0.03820
  unit: samples/sec
- key: latency
  value: 26.18
  unit: s
- key: accuracy
  value: 0.20004
  unit: percentage
```

## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
