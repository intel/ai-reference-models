# SSD-RN34 CPU Inference

SSD-RN34 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/weiliu89/caffe/tree/ssd       |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
* Installation of [PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc and tcmalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

# Prepare Dataset
  Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <MODEL_DIR=path_to_ssd-resnet34_inference_cpu>
export DATASET_DIR=<directory where the dataset will be saved>
./download_dataset.sh
cd -
```

# Download Pretrained Model
cd <MODEL_DIR=path_to_ssd-resnet34_inference_cpu>
export CHECKPOINT_DIR=<directory where to save the pretrained model>
./download_model.sh

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/sdd-resnet34/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

5. Run setup scripts
```
./setup.sh
```
6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)       | `export TEST_MODE=THROUGHPUT (THROUGHPUT, ACCURACY, REALTIME)`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-coco>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <Select from: fp32, avx-fp32, bf16, int8, bf32, or avx-int8>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to an output directory>`                              |
| **CHECKPOINT_DIR**    |                               `export CHECKPOINT_DIR=<path to pre-trained model>`                               |
| **MODEL_DIR** | `export MODEL_DIR=$PWD (set the current path)` |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |
 **WEIGHT_SHARING**(optional)    | `export WEIGHT_SHAREING=False (It is false by default but if you want to run with weight sharing, please set it to true)`   |

7. Run `run_model.sh`


## Output
```
Predicting Ended, total time: 507.41 s
inference latency 73.46 ms
inference performance 13.61 fps
decoding latency 1.03 ms
decoding performance 974.44 fps
Throughput: 13.425 fps
Current AP: 0.20004 AP goal: 0.20000
Accuracy: 0.20004
```


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
