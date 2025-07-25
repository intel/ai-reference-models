# SSD-RN34 CPU Training

SSD-RN34 Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://github.com/weiliu89/caffe/tree/ssd       |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
* Installation of [Build PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Installation of [oneccl-bind-pt](https://pytorch-extension.intel.com/release-whl/stable/cpu/us/oneccl-bind-pt/) (if running distributed)
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
* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can [refer](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
  ```
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

# Prepare Dataset
  Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <MODEL_DIR=path_to_ssd-resnet34_training_cpu>
export DATASET_DIR=<directory where the dataset will be saved>
./download_dataset.sh
cd -
```
# Download Pretrained Model
cd <MODEL_DIR=path_to_ssd-resnet34_training_cpu>
export CHECKPOINT_DIR=<directory where to save the pretrained model>
./download_model.sh

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/sdd-resnet34/training/cpu`
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
| **TEST_MODE** (THROUGHPUT OR ACCURACY)              |                               `export TEST_MODE=THROUGHPUT`                                  |
| **DISTRIBUTED** (leave unset if training single node)              |                               `export DISTRIBUTED=true`                                  |
| **NNODES** (leave unset if training single node)              |                               `export NNODES=2`                                  |
| **HOSTFILE** (leave unset if training single node)              |                               `export HOSTFILE=<your-host-file>`                                  |
| **NUM_RANKS** (leave unset if training single node)              |                               `export NUM_RANKS=1`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-coco>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <Select from: fp32, avx-fp32, bf16, or bf32>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **CHECKPOINT_DIR**    |                               `export CHECKPOINT_DIR=<path to pre-trained model>`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |

7. Run `run_model.sh`

## Output
The training output looks like this:
```
Iteration:     80, Loss function: 9.13430500, Average Loss: 0.98246424
Train: [ 90/100]        TrainTime 10.296 (10.232)
Iteration:     90, Loss function: 8.76533031, Average Loss: 1.06040848
:::MLLOG {"namespace": "", "time_ms": 1719249164799, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "train.py", "lineno": 575, "first_epoch_num": 1, "epoch_count": 70}}
train latency 45.67 ms
train performance 21.90 fps
Throughput: 21.897 fps
```


Final results of the training run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 21.897
  unit: fps
- key: latency
  value: 45.67
  unit: ms
- key: accuracy
  value: 0.20004
  unit: percentage
```
