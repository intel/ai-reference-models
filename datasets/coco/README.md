# COCO Dataset Scripts

## Download and preprocess the COCO validation images

The [COCO dataset](http://cocodataset.org/#home) validation images are
used for inference with object detection models.

The [preprocess_coco_val.sh](preprocess_coco_val.sh) script calls the
[create_coco_tf_record.py](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/dataset_tools/create_coco_tf_record.py)
script from the [TensorFlow Model Garden](https://github.com/tensorflow/models)
to convert the raw images and annotations to TF records.

### Download the raw images and annotations

Prior to running the script, you must download and extract the COCO
validation images and annotations from the
[COCO website](https://cocodataset.org/#download).
```
export DATASET_DIR=<directory where raw images/annotations will be downloaded>
mkdir -p $DATASET_DIR
cd $DATASET_DIR

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Convert the dataset to TF Records

1. Clone the [TensorFlow models repo](https://github.com/tensorflow/models)
   using the specified git SHA and save the directory path to the
   `TF_MODELS_DIR` environment variable.

   | Model | Git SHA |
   |-------|---------|
   | RFCN | `1efe98bb8e8d98bbffc703a90d88df15fc2ce906` |
   | SSD-MobileNet | `7a9934df2afdf95be9405b4e9f1f2480d748dc401` |
   | SSD-ResNet34 | `1efe98bb8e8d98bbffc703a90d88df15fc2ce906` |

   ```
   git clone https://github.com/tensorflow/models.git tensorflow-models
   cd tensorflow-models
   git checkout <Git SHA>
   export TF_MODELS_DIR=$(pwd)
   cd ..
   ```

2. Install the prerequisites mentioned in the
   [TensorFlow models object detection installation doc](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#dependencies)
   and run [protobuf compilation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
   on the code that was cloned in the previous step.

3. Download and run the [preprocess_coco_val.sh](preprocess_coco_val.sh)
   script, uses code from the TensorFlow models repo to convert the
   validation images to the TF records format. At this point, you should
   already have the `TF_MODELS_DIR` path set from step one of this
   section and the `DATASET_DIR` set to the location where raw images
   and annotations were downloaded. Set the `OUTPUT_DIR` variable to
   point to the location where the TF records file will be written, then
   run the script.
   ```
   export OUTPUT_DIR=<directory where TF records will be written>

   ./preprocess_coco_val.sh
   ```

   After the script completes, the `OUTPUT_DIR` will have a TF records file
   for the coco validation dataset:
   ```
   $ ls $OUTPUT_DIR
   coco_val.record
   ```
