# COCO Dataset Scripts

## Download and preprocess the COCO validation images

The [COCO dataset](http://cocodataset.org/#home) validation images are used
for inference with object detection models.

The [preprocess_coco_val.sh](preprocess_coco_val.sh) script calls the
[create_coco_tf_record.py](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/dataset_tools/create_coco_tf_record.py)
script from the [TensorFlow Model Garden](https://github.com/tensorflow/models)
to convert the raw images and annotations to TF records. The version of
the conversion script that you will need to use will depend on which
model is being run. The table below has git commit ids for the
[TensorFlow Model Garden](https://github.com/tensorflow/models) that have
been tested with each model.

| Model | Git Commit ID |
|-------|---------|
| Faster R-CNN | `7a9934df2afdf95be9405b4e9f1f2480d748dc40` |
| RFCN | `1efe98bb8e8d98bbffc703a90d88df15fc2ce906` |
| SSD-MobileNet | `7a9934df2afdf95be9405b4e9f1f2480d748dc40` |
| SSD-ResNet34 | `1efe98bb8e8d98bbffc703a90d88df15fc2ce906` |

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

Set following environment variables are expected by the script:
* `DATASET_DIR`: Parent directory of the val2017 raw images and annotations files
* `OUTPUT_DIR`: Directory where the TF records file will be written

Follow the instructions below to run the script on
[bare metal](#bare-metal) or [in a docker container](#docker), if the
model that you are running requires the dataset to be in the TF records
format.

### Bare Metal

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

2. Install the prerequisites mentioned in the
   [TensorFlow models object detection installation doc](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#dependencies)
   and run [protobuf compilation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
   on the code that was cloned in the previous step.

3. Download and run the [preprocess_coco_val.sh](preprocess_coco_val.sh)
   script, which uses code from the TensorFlow models repo to convert the
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

### Docker

1. The container used in the command below includes the prerequisites
   needed to run the dataset preprocessing script. You will need to
   mount volumes for the dataset (raw images and annotations) and the
   output dirctory (the location where the TF records file will be
   written), and set the `TF_MODELS_BRANCH` environment variable to the
   git commit id for the
   [TensorFlow Model Garden](https://github.com/tensorflow/models).

   ```
   export DATASET_DIR=<Parent directory of the val2017 raw images and annotations files>
   export OUTPUT_DIR=<directory where TF records will be written>
   export TF_MODELS_BRANCH=<git commit id>

   docker run \
   --env VAL_IMAGE_DIR=${DATASET_DIR}/val2017 \
   --env ANNOTATIONS_DIR=${DATASET_DIR}/annotations \
   --env TF_MODELS_BRANCH=${TF_MODELS_BRANCH} \
   --env OUTPUT_DIR=${OUTPUT_DIR} \
   -v ${DATASET_DIR}:${DATASET_DIR} \
   -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
   -t intel/object-detection:tf-1.15.2-imz-2.2.0-preprocess-coco-val
   ```

   After the script completes, the `OUTPUT_DIR` will have a TF records file
   for the coco validation dataset:
   ```
   $ ls $OUTPUT_DIR
   coco_val.record
   ```
