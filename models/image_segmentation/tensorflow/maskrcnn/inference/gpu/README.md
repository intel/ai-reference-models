# Mask RCNN Inference

Mask RCNN Inference BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** 
| :---: | :---: | :---: | :---: |:----------:|
|   training   |  Tensorflow   | [DeepLearningExamples/MaskRCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |        5be8a3cae21ee2d80e3935a4746827cb3367bcac         |  [EnableInference.patch](#EnableInference.patch)       |

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Dataset 
Download & preprocess COCO 2017 dataset. 
```
export DATASET_DIR=/path/to/dataset/dir
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/dataset
bash download_and_preprocess_coco.sh $DATASET_DIR
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models/image_segmentation/tensorflow/maskrcnn/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Download weights
    ```
        pushd .
        cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
        python scripts/download_weights.py --save_dir=./weights
        popd 
    ```
5. Setup required environment paramaters

    |   **Parameter**    |                   **export command**                   |
    |:------------------------------------------------------:| :--- |
    |  **DATASET_DIR**   |       `export DATASET_DIR=/the/path/to/dataset`        |
    |   **PRETRAINED_DIR**   | `export PRETRAINED_DIR=/the/path/to/pretrained_model/` |
    |   **PRECISION**   |         `export PRECISION=fp16` (fp16 or fp32)         |
    |   **BATCH_SIZE** (optional)  |                 `export BATCH_SIZE=16`                 |
6. Run `run_model.sh`

## Output

Output will typicall looks like:
```
2023-09-15 12:45:12.198281: I tensorflow/core/framework/local_rendezvous.cc:421] Local rendezvous recv item cancelled. Key hash: 15253968942006736824
() {'predict_throughput': 32.896633477338334, 'predict_latency': 0.4863719569062287, 'predict_latency_90': 0.48666301561606345, 'predict_latency_95': 0.48671875026262756, 'predict_latency_99': 0.48682774246035293, 'predict_time': 166.06602716445923}
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 32.896633477338334
   unit: images/sec
```
