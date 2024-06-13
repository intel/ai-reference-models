# BERT Large Training

BERT Large training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch     |    [NVIDIA/DeepLearningExamples](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)        |           main          |        -       |

# Pre-Requisite
* Host has one of the following GPUs:
  * **Arc Series** - [Intel® Arc™ A-Series Graphics](https://ark.intel.com/content/www/us/en/ark/products/series/227957/intel-arc-a-series-graphics.html)
  * **Max Series** - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max & Arc Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.


# Prepare Dataset
## Dataset: 
Please refer to [NV-bert repository](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09#download-and-prepare-the-data) to download and prepare the data.

The dataset should be like below:
```
|_hdf5  
      |_ eval                               # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data preparation)  
      |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length *used for training*
      |_ training                           # 500 chunks in binary hdf5 format 
      |_ training_4320                      # 
         |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length (not used in training, can delete after data   preparation)
         |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length *used for training*
```
We are using  `hdf5/hdf5_4320_shards_varlength` as the our dataset.

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/bert_large/training/gpu`
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
6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                    |
|:----------------------------:|:--------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100 or Arc Series GPU) |
| **PLATFORM**                 | `export PLATFORM=Max` (Max or Arc)                                                     |
| **NUM_DEVICES**              | `export NUM_DEVICES=<num_devices>` (`<num_devices>` is the number of GPU devices used for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so `<num_devices>` can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU or Arc Series GPU, the number of GPU devices available in each node is the same as number of GPUs, so `<num_devices>` can be set as <=8 for a node with 8 single-tile GPUs.) |
| **DATASET_DIR**              | `export DATASET_DIR=</the/path/to/dataset>`                                            |
| **OUTPUT_DIR**               | `export OUTPUT_DIR=</the/path/to/output_dir>`                                          |
| **BATCH_SIZE** (optional)    | `export BATCH_SIZE=16`                                                                 |
| **PRECISION** (optional)     | `export PRECISION=BF16` (BF16 FP8 FP32 and TF32 are supported for Max and BF16 for Arc)|
|**NUM_ITERATIONS** (optional) | `export NUM_ITERATIONS=20`                                                             |

7. Run `run_model.sh`

## Output

Single-device output will typically look like:

```
[info] construct file from initialization
[info] input dir =  /home/dataset/hdf5/hdf5_4320_shards_varlength
[info] num files =  4282
epoch: 1
Loaded 193485 samples from datafile: /home/dataset/hdf5/hdf5_4320_shards_varlength/pretrain-part-01.hdf5
bert_train latency:  0.24147300720214843  s
bert_train throughput:  66.25999396531161  sentences/s
perplexity = 11.020857810974121
```

Multi-device output will typically look like:
```
Model to device:  xpu:0
using adamw
Doing torch xpu optimize, dtype:  torch.bfloat16
Torch distributed is available.
Torch distributed is initialized.
[info] Setting seed:  123 . worker seed:  224899942


found num checkpoints:  0
resume from checkpoints:  False
resume checkpoint:  None
[info] construct file from initialization
[info] input dir =  /home/dataset/hdf5/hdf5_4320_shards_varlength
[info] num files =  4282
epoch: 1
Loaded 194779 samples from datafile: /home/dataset/hdf5/hdf5_4320_shards_varlength/pretrain-part-00.hdf5
bert_train latency:  0.2703933477401733  s
bert_train throughput:  59.17305338212218  sentences/s
perplexity = 11.018452644348145
Setting seed to ensure same model master weight at the beginning.
world_size:2, rank:1, device:xpu:1
args.world_size=2, args.rank=1
Get config from config_name bert_config.json
Set different weight_decay for model parameters
GroupSizes:  [335869952, 356156]
Model to device:  xpu:1
using adamw
Doing torch xpu optimize, dtype:  torch.bfloat16
Torch distributed is available.
Torch distributed is initialized.
[info] Setting seed:  123 . worker seed:  1749090055


found num checkpoints:  0
resume from checkpoints:  False
resume checkpoint:  None
[info] construct file from initialization
[info] input dir =  /home/dataset/hdf5/hdf5_4320_shards_varlength
[info] num files =  4282
Loaded 193485 samples from datafile: /home/dataset/hdf5/hdf5_4320_shards_varlength/pretrain-part-01.hdf5
bert_train latency:  0.2730390548706055  s
bert_train throughput:  58.599675447831  sentences/s
perplexity = 11.12635612487793
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 66.259994
   unit: sent/s
 - key: latency
   value: 0.2414730072021484
   unit: s
 - key: accuracy
   value: 11.021
   unit: perplexity
```
