# Bert-Large Model Training

Bert-Large Model Training using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** |          **Model Repo**           | **Branch/Commit/Tag** |**Optional Patch** |
| :---: | :---: |:---------------------------------:| :---: | :---: |
|   training   |  TensorFlow   | [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) |        master         | [bert-large-itex-bf16.patch](#bert-large-itex-bf16.patch) |

# Pre-Requisite

* Host has Intel® Data Center GPU Max.
* Host has installed latest Intel® Data Center GPU Max Series Driver https://dgpu-docs.intel.com/driver/installation.html

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
6. If you want to run with Horovod in multi-tile (export MULTI_TILE=True), please install intel-optimization-for-horovod instead of horovod.
`pip uninstall horovod && pip install intel-optimization-for-horovod`
6. Setup required environment paramaters

   | **Parameter**                  |              **export command**              |
   |:--------------------------------------------:|:----------------------------------------------:|
   | **DATA_DIR**                |    `export DATA_DIR=/the/path/to/dataset`    |
   | **RESULTS_DIR**                 | `export RESULTS_DIR=/the/path/to/result_dir` |
   | **DATATYPE**      | `export DATATYPE=bf16` (bf16, tf32 or fp32)  |
   | **MULTI_TILE**           |  `export MULTI_TILE=False` (False or True)   |

7. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

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

Multi-tile output will typicall looks like:

```
[0] I1030 10:59:04.732321 139681571383104 model_training_utils.py:83] Training Summary: 
[0] {'total_training_steps': 8, 'train_loss': 10.069790840148926}
[0] I1030 10:59:04.732495 139681571383104 model_training_utils.py:595] -----------------------------
[0] I1030 10:59:04.732545 139681571383104 model_training_utils.py:596]   Batch size = 32
[0] I1030 10:59:04.732575 139681571383104 model_training_utils.py:597]   Num steps = 8
[0] I1030 10:59:04.732604 139681571383104 model_training_utils.py:598]   LR = 0.0005
[0] I1030 10:59:04.732630 139681571383104 model_training_utils.py:600] Multi-GPU training with TF Horovod
[0] I1030 10:59:04.732667 139681571383104 model_training_utils.py:601] hvd.size() = 2
[0] I1030 10:59:04.732693 139681571383104 model_training_utils.py:602] Total Training Time = 279.63 for Sequences = 15360
[0] I1030 10:59:04.732739 139681571383104 model_training_utils.py:604] Throughput Average (sequences/sec) with overhead = 54.93
[0] I1030 10:59:04.732893 139681571383104 model_training_utils.py:606] Throughput Average (sequences/sec) = 197.41
[0] I1030 10:59:04.732919 139681571383104 model_training_utils.py:607] -----------------------------
[0] decayed_learning_rate_at_crossover_point = 1.000000e-03, adjusted_init_lr = 1.000000e-03
[0] DLL 2023-10-30 10:59:04.732944 -  throughput_train : 197.408 sequences/s
[0] DLL 2023-10-30 10:59:04.733007 -  total_loss : 10.0698 
[0] 2023:10:30-10:59:08:(273540) |CCL_INFO| finalizing level-zero
[0] 2023:10:30-10:59:08:(273540) |CCL_INFO| finalized level-zero
```

Final results of the training run can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 99.048
   unit: sequences/s
```
