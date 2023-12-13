# Mask RCNN training

Mask RCNN Training using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** |      **Branch/Commit/Tag**      | **Optional Patch** |
| :---: | :---: | :---: |:-------------------------------:|:--------:|
|   training   |  TensorFlow   | [DeepLearningExamples/MaskRCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |         master                  | [EnableBF16.patch](#bf16patch) |

# Pre-Requisite

* Host has Intel® Data Center GPU Max
* Host has installed latest Intel® Data Center GPU Max Series
  Driver https://dgpu-docs.intel.com/driver/installation.html

# Dataset

Download & preprocess COCO 2017 dataset.

```
export DATASET_DIR=/path/to/dataset/dir
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/dataset
bash dataset/download_and_preprocess_coco.sh $DATASET_DIR
```

## Training

1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/maskrcnn/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. If you run multi-tile (export MULTI_TILE=True), please install "intel-optimization-for-horovod" with `pip install intel-optimization-for-horovod`
6. Download weights
    ```
        pushd .
        cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
        python scripts/download_weights.py --save_dir=./weights
        popd 
    ```
7. Setup required environment paramaters

   | **Parameter**                  |               **export command**                |
   |:-----------------------------------------------:|:----------------------------------------------:|
   | **DATASET_DIR**                |    `export DATASET_DIR=/the/path/to/dataset`    |
   | **OUTPUT_DIR**  (optional)                 |   `export OUTPUT_DIR=/the/path/to/output_dir`   |
   | **BATCH_SIZE** (optional)      |              `export BATCH_SIZE=4`              |
   | **PRECISION**                  | `export PRECISION=bfloat16` (bfloat16 or fp32)  |
   | **EPOCHS** (optional)          |                `export EPOCHS=1`                |
   | **STEPS_PER_EPOCH** (optional) |           `export STEPS_PER_EPOCH=20`           |
   | **MULTI_TILE**           |    `export MULTI_TILE=False` (False or True)    |

8. Run `run_model.sh`

## Output

Single-tile output will typically look like:

```
2023-09-11 14:54:49,905 I dllogger        (1, 20) loss: 639.5632934570312
2023-09-11 14:54:49,906 I dllogger        (1, 20) train_time: 23.89216899871826, train_throughput: 21.438093303125907
2023-09-11 14:54:49,914 I dllogger        (1,) loss: 639.5632934570312
2023-09-11 14:54:49,914 I dllogger        () loss: 639.5632934570312
2023-09-11 14:54:49,915 I dllogger        () train_time: 23.90118169784546, train_throughput: 23.507529269636105
```
Multi-tile output will typically look like:

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
 - key: throughput
   value: 23.507529269636105
   unit: images/sec
```
