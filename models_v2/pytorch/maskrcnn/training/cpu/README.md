# MaskRCNN CPU Training

MaskRCNN Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://github.com/matterport/Mask_RCNN        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
* Installation of [Build PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Installation of [oneccl-bind-pt](https://pytorch-extension.intel.com/release-whl/stable/cpu/us/oneccl-bind-pt/) (if running distributed)
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
* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
  ```bash
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

# Prepare Dataset
  Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <MODEL_DIR=path_to_maskrcnn_training_cpu>
export DATASET_DIR=<directory where the dataset will be saved>
./download_dataset.sh
cd -
```

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/maskrcnn/training/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

5. Run setup scripts
```
cd <MODEL_DIR=path/to/maskrcnn/training/cpu>
./setup.sh
cd <path/to/maskrcnn/training/cpu/maskrcnn-benchmark>
pip install -e setup.py develop
pip install -r requirements.txt
cd -
```
6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MODEL_DIR**    |                               `export MODEL_DIR=$PWD`                               |
| **DISTRIBUTED** (leave unset if single node)              |                               `export DISTRIBUTED=true`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-coco>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <Select from: fp32, avx-fp32, bf16, or bf32>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **NNODES** (only if distributed=true)   |     `export NNODES=#your_node_number`   |
| **HOSTFILE** (only if distributed=true)   |   ` export HOSTFILE=your_ip_list_file #one ip per line`  |
| **LOCAL_BATCH_SIZE** (only if distributed=true)  |  `export LOCAL_BATCH_SIZE=<set local batch size>`  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |

7. Run `run_model.sh`
## Output


```
[1] 2023-10-28 06:00:28,866 I dllogger        (1, 20) train_time: 38.43677306175232, train_throughput: 27.085276273687406
[0] 2023-10-28 06:00:28,866 I dllogger        (1, 20) train_time: 38.44535684585571, train_throughput: 27.104091813787576
[1] 2023-10-28 06:00:28,870 I dllogger        (1,) loss: 694.025390625
[1] 2023-10-28 06:00:28,870 I dllogger        () loss: 694.025390625
[0] 2023-10-28 06:00:28,870 I dllogger        (1,) loss: 694.025390625
[1] 2023-10-28 06:00:28,870 I dllogger        () train_time: 38.44074249267578, train_throughput: 27.125991704955165
[0] 2023-10-28 06:00:28,870 I dllogger        () loss: 694.025390625
[0] 2023-10-28 06:00:28,870 I dllogger        () train_time: 38.449461460113525, train_throughput: 27.118943899083977
```


Final results of the training run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 27.118943899083977
  unit: fps
- key: latency
  value: 10605.99
  unit: ms
- key: accuracy
  value: NA
  unit: percentage
```
