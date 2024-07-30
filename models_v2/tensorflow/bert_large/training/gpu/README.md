# Bert-Large Model Training

Bert-Large Model Training using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** |          **Model Repo**           | **Branch/Commit/Tag** |**Optional Patch** |
| :---: | :---: |:---------------------------------:| :---: | :---: |
|   training   |  TensorFlow   | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) |        master         | [bert-large-itex-bf16.patch](#bert-large-itex-bf16.patch) |

**Note**: Refer to [CONTAINER.md](CONTAINER.md) for BERT-Large training instructions using docker containers.
# Pre-Requisite

* Host has Intel® Data Center GPU Max.
* Host has installed latest Intel® Data Center GPU Max Series Driver https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)

  Recommend to use a package manager like apt, yum or dnf to install the packages above. Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset

 * Prapare total dataset from scratch. [NV-bert repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT)  provides scripts to download, verify, and extract the SQuAD dataset and pretrained weights for fine-tuning as well as Wikipedia and BookCorpus dataset for pre-training.

 * You can run below scripts to download datasets for fine-tuning and pretraining. Remember to modify the environment varible ```BERT_PREP_WORKING_DIR ``` in ```data/create_datasets_from_start.sh``` to your real data dir.
   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/TensorFlow2/LanguageModeling/BERT
   bash data/create_datasets_from_start.sh pretrained
   ```
 * Then use below scripts and patch to create small dataset for bert large training performance.
    ```
    git clone https://github.com/IntelAI/models.git
    cp models/models_v2/tensorflow/bert_large/training/gpu/bert-large-itex-bf16.patch .
    git am --signoff < bert-large-itex-bf16.patch
    bash data/create_small_dataset_for_benchmark.sh [dataset_output_dir](default: ${PWD}/dataset)
    ```
    The dataset structure will be:

  + [dataset_output_dir]
     - training/
     - test/
     - uncased_L-24_H-1024_A-16/

## Run Model

1. If not already cloned, clone Intel® AI Reference Model repository
`git clone https://github.com/IntelAI/models.git` 
2. `cd models/models_v2/tensorflow/bert_large/training/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Setup required environment paramaters

   | **Parameter**                  |              **export command**              |
   |:--------------------------------------------:|:----------------------------------------------:|
   | **DATA_DIR**                |    `export DATA_DIR=/the/path/to/dataset`    |
   | **RESULTS_DIR**                 | `export RESULTS_DIR=/the/path/to/result_dir` |
   | **DATATYPE**      | `export DATATYPE=bf16` (bf16, tf32 or fp32)  |
   | **MULTI_TILE**           |  `export MULTI_TILE=False` (provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100)   |
   | **NUM_DEVICES**          |  `export NUM_DEVICES=<num_devices>` (`<num_devices>` is the number of GPU devices to use for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so `<num_devices>` can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU, the number of GPU devices available in each node is the same as number of GPUs, so `<num_devices>` can be set as <=8 for a node with 8 Max 1100 GPUs.)     | 

7. Run `run_model.sh`

## Output

Single-device output will typically look like:

```
I0907 03:06:10.519348 140707593221952 model_training_utils.py:83] Training Summary: 
{'total_training_steps': 8, 'train_loss': 10.389582633972168}
I0907 03:06:10.519572 140707593221952 model_training_utils.py:595] -----------------------------
I0907 03:06:10.519612 140707593221952 model_training_utils.py:596]   Batch size = 32
I0907 03:06:10.519635 140707593221952 model_training_utils.py:597]   Num steps = 8
I0907 03:06:10.519656 140707593221952 model_training_utils.py:598]   LR = 0.0005
I0907 03:06:10.519677 140707593221952 model_training_utils.py:602] Total Training Time = 339.25 for Sequences = 7680
I0907 03:06:10.519717 140707593221952 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = 22.64
I0907 03:06:10.520012 140707593221952 model_training_utils.py:606] Throughput Average (sequences/sec) = 99.05
I0907 03:06:10.520030 140707593221952 model_training_utils.py:607] -----------------------------
decayed_learning_rate_at_crossover_point = 5.000000e-04, adjusted_init_lr = 5.000000e-04
DLL 2023-09-07 03:06:10.520066 -  throughput_train : 99.048 sequences/s
DLL 2023-09-07 03:06:10.520158 -  total_loss : 10.3896 
```

Multi-device output will typically look like:

```
I1030 10:59:04.732321 139681571383104 model_training_utils.py:83] Training Summary: 
{'total_training_steps': 8, 'train_loss': 10.069790840148926}
I1030 10:59:04.732495 139681571383104 model_training_utils.py:595] -----------------------------
I1030 10:59:04.732545 139681571383104 model_training_utils.py:596]   Batch size = 32
I1030 10:59:04.732575 139681571383104 model_training_utils.py:597]   Num steps = 8
I1030 10:59:04.732604 139681571383104 model_training_utils.py:598]   LR = 0.0005
I1030 10:59:04.732630 139681571383104 model_training_utils.py:600] Multi-GPU training with TF Horovod
I1030 10:59:04.732667 139681571383104 model_training_utils.py:601] hvd.size() = 2
I1030 10:59:04.732693 139681571383104 model_training_utils.py:602] Total Training Time = 279.63 for Sequences = 15360
I1030 10:59:04.732739 139681571383104 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = 54.93
I1030 10:59:04.732893 139681571383104 model_training_utils.py:606] Throughput Average (sequences/sec) = 197.41
I1030 10:59:04.732919 139681571383104 model_training_utils.py:607] -----------------------------
decayed_learning_rate_at_crossover_point = 1.000000e-03, adjusted_init_lr = 1.000000e-03
DLL 2023-10-30 10:59:04.732944 -  throughput_train : 197.408 sequences/s
DLL 2023-10-30 10:59:04.733007 -  total_loss : 10.0698 
2023:10:30-10:59:08:(273540) |CCL_INFO| finalizing level-zero
2023:10:30-10:59:08:(273540) |CCL_INFO| finalized level-zero
```

Final results of the training can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 99.048
   unit: sequences/s
```
