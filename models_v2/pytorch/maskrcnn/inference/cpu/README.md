# MaskRCNN CPU Inference

MaskRCNN Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/matterport/Mask_RCNN        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
* Installation of [Build PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc and tcmalloc should be built from the [General](#general-setup) setup section.
  ```bash
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
cd <MODEL_DIR=path_to_maskrcnn_inference_cpu>
export DATASET_DIR=<directory where the dataset will be saved>
./download_dataset.sh
cd -
```

# Download pretrained model
```
cd <MODEL_DIR=path_to_maskrcnn_inference_cpu>
export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
./download_model.sh
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/maskrcnn/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

5. Run setup scripts
```
cd <MODEL_DIR=path_to_maskrcnn/inference/cpu>
./setup.sh
cd <path/to/maskrcnn/inference/cpu/maskrcnn-benchmark>
pip install -e setup.py develop
pip install -r requirements.txt
cd -
```
6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)       | `export TEST_MODE=THROUGHPUT (THROUGHPUT, ACCURACY, REALTIME)`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-coco>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <Select from: fp32, avx-fp32, bf16, or bf32>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **CHECKPOINT_DIR**    |                               `export CHECKPOINT_DIR=<path to pre-trained model>`                               |
| **MODE**    |                               `export MODE=<set to 'jit' or 'imperative'>`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |

7. Run `run_model.sh`
## Output


```
2024-05-06 15:01:22,842 maskrcnn_benchmark.inference INFO: P99 Latency 10605.99 ms
2024-05-06 15:01:22,842 - maskrcnn_benchmark.inference - INFO - P99 Latency 10605.99 ms
2024-05-06 15:01:22,843 maskrcnn_benchmark.inference INFO: Total run time: 0:06:53.501260 (20.67506300210953 s / iter per device, on 1 devices)
2024-05-06 15:01:22,843 - maskrcnn_benchmark.inference - INFO - Total run time: 0:06:53.501260 (20.67506300210953 s / iter per device, on 1 devices)
2024-05-06 15:01:22,843 maskrcnn_benchmark.inference INFO: Model inference time: 0:03:27.253329 (10.36266644001007 s / iter per device, on 1 devices)
2024-05-06 15:01:22,843 - maskrcnn_benchmark.inference - INFO - Model inference time: 0:03:27.253329 (10.36266644001007 s / iter per device, on 1 devices)
Throughput: 5.404 fps
```


Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 5.404
  unit: fps
- key: latency
  value: 10605.99
  unit: ms
- key: bounding-box accuracy
  value: 0.381
  unit: percentage
- key: segmentation accuracy
  value: 0.381
  unit: percentage
```
