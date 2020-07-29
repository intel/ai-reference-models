# COCO Dataset Scripts

## Download and preprocess the COCO validation images

The [COCO dataset](http://cocodataset.org/#home) validation images are used
for inference with object detection models.

The [preprocess_coco_val.sh](preprocess_coco_val.sh) script calls the
[create_coco_tf_record.py](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/dataset_tools/create_coco_tf_record.py)
script from the [TensorFlow Model Garden](https://github.com/tensorflow/models)
to convert the raw images and annotations to TF records.

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

If the script is not being run in the model zoo's container, the `TF_MODELS_DIR`
environment variable will need to be set to point to a clone of
the [TensorFlow Model Garden](https://github.com/tensorflow/models) repo
and the [dependencies needed to run object detection](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/g3doc/installation.md#installation)
need to be installed in your environment.

The snipped below shows how to run the coco preprocessing container,
which mounts input and output directories and then runs the script to
create TF records in the output directory.
```
export DATASET_DIR=<Parent directory of the val2017 raw images and annotations files>
export OUTPUT_DIR=<directory where TF records will be written>

docker run \
--env VAL_IMAGE_DIR=${DATASET_DIR}/val2017 \
--env ANNOTATIONS_DIR=${DATASET_DIR}/annotations \
--env OUTPUT_DIR=${OUTPUT_DIR} \
-v ${DATASET_DIR}:${DATASET_DIR} \
-v ${OUTPUT_DIR}:${OUTPUT_DIR} \
-t amr-registry.caas.intel.com/aipg-tf/model-zoo:1.15.2-object-detection-preprocess-coco-val
```

After the script completes, the `OUTPUT_DIR` will have a TF records file
for the coco validation dataset:
```
$ ls $OUTPUT_DIR
coco_val.record
```
