# PyTorch YOLOv7 inference

## Description

This document has instructions for running YOLOv7 inference using
Intel-optimized PyTorch and PyTorch inductor.

## Model Information

| Use Case    | Framework   | Model Repository| Branch/Commit| Patch |
|-------------|-------------|-----------------|--------------|--------------|
| Inference   | Pytorch     | https://github.com/WongKinYiu/yolov7 | main/a207844 | [`yolov7_ipex_and_inductor.patch`](/models_v2/pytorch/yolov7/inference/cpu/yolov7_ipex_and_inductor.patch). Enable yolov7 inference with IPEX and torch inductor for specified precision (fp32, int8, bf16, fp16, or bf32). |

## Pre-Requisite
Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniforge and build Pytorch, IPEX, TorchVison and Tcmalloc.

* Set Tcmalloc Preload for better performance

  After [Tcmalloc setup](/docs/general/pytorch/BareMetalSetup.md#build-tcmalloc), set the following environment variables.
  ```
  export LD_PRELOAD="<path to the tcmalloc directory>/lib/libtcmalloc.so":$LD_PRELOAD
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env. Set the following environment variables.
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

## Model specific instructions
* If you are running fp16 on a platform supporting AMX-FP16, add the following environment variables to leverage AMX-FP16 for better performance.

```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* If you are running bf16/fp16/int8 on a platform supporting AVX2-VNNI-2, add the following environment variables to leverage AVX2-VNNI-2 for better performance.

```
    export DNNL_MAX_CPU_ISA=AVX2_VNNI_2
```
For more detailed information about `DNNL_MAX_CPU_ISA` of oneDNN, see oneDNN [link](https://oneapi-src.github.io/oneDNN/index.html).


## Download pretrained model
Export the `CHECKPOINT_DIR` environment variable to specify the directory where the pretrained model
will be saved. This environment variable will also be used when running quickstart scripts.
```
cd <clone of AIRM/models_v2/pytorch/yolov7/inference/cpu>
export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
chmod a+x *.sh
./download_model.sh
```

## Prepare Dataset
Prepare the 2017 [COCO dataset](https://cocodataset.org) for yolov7 using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be saved. This environment variable will also be used when running quickstart scripts.
```
cd <clone of AIRM/models_v2/pytorch/yolov7/inference/cpu>
export DATASET_DIR=<directory where the dataset will be saved>
./download_dataset.sh
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/yolov7/inference/cpu`
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
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-dlrm_data> or <path-to-preprocessed-data>`                                  |
| **CHECKPOINT_DIR**      |                 `export CHECKPOINT_DIR=<directory where the pretrained model will be saved>`        |
| **PRECISION**    |                               `export PRECISION=fp32 <specify the precision to run: fp32, int8, bf32, bf16, or fp16>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to the directory where log files will be written>`                               |
| **MODEL_DIR** | `export MODEL_DIR=$PWD (set the current path)` |
| **BATCH_SIZE** (optional)  |                        `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |
| **TORCH_INDUCTOR** (optional)    | `export TORCH_INDUCTOR=< 0 or 1> (Compile model with PyTorch Inductor backend)`   |

8. Run `run_model.sh`

## Output
Output typically looks like this:
100%|██████████| 4/4 [00:17<00:00,  4.29s/it]
time per prompt(s): 73.46
Inference latency  73.46 ms
Throughput: 13.61 fps

Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 13.61
  unit: fps
- key: latency
  value: 73.46
  unit: ms
- key: accuracy
  value: 0.20004
  unit: percentage
```

## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
