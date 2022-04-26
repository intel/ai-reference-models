# COCO Training Dataset

## Download and preprocess the COCO train images on bare-metal

The [COCO dataset](http://cocodataset.org/#home) train images are used
for Training with object detection models.

The [preprocess_coco_training.sh](preprocess_coco_training.sh) script calls the
[create_coco_tf_record.py](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/dataset_tools/create_coco_tf_record.py)
script from the [TensorFlow Model Garden](https://github.com/tensorflow/models)
to convert the raw images and annotations to TF records. The version of
the conversion script that you will need to use will depend on which
model is being run. The table below has git commit ids for the
[TensorFlow Model Garden](https://github.com/tensorflow/models) that have
been tested with each model.

| Model | Git Commit ID |
|-------|-------------- |
| SSD-ResNet34 | `1efe98bb8e8d98bbffc703a90d88df15fc2ce906` |

Prior to running the script, you must download and extract the COCO
train images and annotations from the
[COCO website](https://cocodataset.org/#download).
```
export DATASET_DIR=<directory where raw images/annotations will be downloaded>
mkdir -p $DATASET_DIR
cd $DATASET_DIR

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

1. Clone the [TensorFlow models repo](https://github.com/tensorflow/models)
   using the git commit id from the table above and save the directory path to the
   `TF_MODELS_DIR` environment variable.

   ```
   git clone https://github.com/tensorflow/models.git tensorflow-models
   cd tensorflow-models
   git checkout <Git commit id>
   export TF_MODELS_DIR=$(pwd)
   cd ..
   ```

2. Install the prerequisites based on the
   [TensorFlow models object detection installation doc](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#dependencies)
   and run [protobuf compilation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
   on the code that was cloned in the previous step.
   ```
   virtualenv --python=python3.6 coco_env
   . coco_env/bin/activate
   
   # Running next command requires root privileges
   apt-get update && apt-get install protobuf-compiler python-pil python-lxml python-tk
   pip install intel-tensorflow==1.15.2
   pip install pycocotools==2.0.2
   
   # Protobuf Compilation, from ${TF_MODELS_DIR}/research directory
   cd ${TF_MODELS_DIR}/research
   protoc object_detection/protos/*.proto --python_out=.
   ```
   Please see the [Manual protobuf-compiler installation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage)
   in case of any errors while compiling.

3. Download and run the [preprocess_coco_train.sh](preprocess_coco_train.sh)
   script, which uses code from the TensorFlow models repo to convert the
   train images to the TF records format. At this point, you should
   already have the `TF_MODELS_DIR` path set from step one of this
   section and the `DATASET_DIR` set to the location where raw images
   and annotations were downloaded. The output TF records file will be written in `DATASET_DIR`, then
   run the script.
   ```
   wget https://raw.githubusercontent.com/IntelAI/models/master/datasets/coco/preprocess_coco_train.sh
   bash preprocess_coco_train.sh
   ```

   After the script completes, the `DATASET_DIR` will have a TF records files `coco_train.record-00000-of-00100`
   for the coco training dataset:
   ```
   $ ls $DATASET_DIR
   annotations
   annotations_trainval2017.zip
   coco_train.record-00000-of-00100
   train2017
   train2017.zip
   ```
 