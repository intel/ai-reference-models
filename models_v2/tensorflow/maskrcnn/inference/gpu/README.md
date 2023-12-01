# Mask RCNN training

Mask RCNN Training BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Weight** | **Optional Patch** |
| :---: | :---: | :---: | :---: | :---: | :---: |
|   training   |  Tensorflow   | [DeepLearningExamples/MaskRCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |        master         | See Section [Prerequisites](#weight) | [EnableInference.patch](#inference patch) |

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

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/maskrcnn/inference/gpu`
3. create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Download weights
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
    |   **PRETRAINED_DIR**   | `export PRETRAINED_DIR=/the/path/to/pretrained_dir`           |
    |   **BATCH_SIZE** (optional)   | `export BATCH_SIZE=4`           |
    |   **PRECISION**   | `export PRECISION=bfloat16` (float16 or fp32)           |
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
   value: xxxx
   unit: images/sec
```
