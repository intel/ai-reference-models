# Mask RCNN training

Mask RCNN Training BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Weight** | **Optional Patch** |
| :---: | :---: | :---: | :---: | :---: | :---: |
|   training   |  Tensorflow   | [DeepLearningExamples/MaskRCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |        master         | See Section [Prerequisites](#weight) | [EnableBF16.patch](#bf16patch) |

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

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
2. `cd models/models/image_segmentation/tensorflow/maskrcnn/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Download weights
    ```
        pushd .
        cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
        python scripts/download_weights.py --save_dir=./weights
        popd 
    ```
5. Setup required environment paramaters

    |   **Parameter**    | **export command**                                    |
    | :---: | :--- |
    |  **DATASET_DIR**   | `export DATASET_DIR=/the/path/to/dataset`             |
    |   **OUTPUT_DIR**   | `export OUTPUT_DIR=/the/path/to/output_dir`           |
    |   **BATCH_SIZE** (optional)   | `export BATCH_SIZE=4`           |
    |   **PRECISION**   | `export PRECISION=bfloat16` (bfloat16 or fp32)           |
    |   **EPOCHS** (optional)  | `export EPOCHS=1`           |
    |   **STEPS_PER_EPOCH** (optional)  | `export STEPS_PER_EPOCH=20`           |
6. Run `run_model.sh`

## Output

Output will typicall looks like:
```
2023-09-11 14:54:49,905 I dllogger        (1, 20) loss: 639.5632934570312
2023-09-11 14:54:49,906 I dllogger        (1, 20) train_time: 23.89216899871826, train_throughput: 21.438093303125907
2023-09-11 14:54:49,914 I dllogger        (1,) loss: 639.5632934570312
2023-09-11 14:54:49,914 I dllogger        () loss: 639.5632934570312
2023-09-11 14:54:49,915 I dllogger        () train_time: 23.90118169784546, train_throughput: 23.507529269636105
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 23.507529269636105
   unit: images/sec
```
