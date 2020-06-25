# COCO Dataset Scripts

## Download and preprocess the COCO validation images

The [COCO dataset](http://cocodataset.org/#home) validation images are used
for inference with object detection models.

The [download_and_preprocess_coco_val.sh](download_and_preprocess_coco_val.sh)
script downloads the raw validation images and annotations and then
calls the [create_coco_tf_record.py](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/dataset_tools/create_coco_tf_record.py)
script from the [TensorFlow Model Garden](https://github.com/tensorflow/models)
to convert the raw images to TF records.

Running the script requires that the `DATASET_DIR` is set to specify
a directory where the raw images and TF records will be written. If the
script is not being run in the model zoo's container, the `TF_MODELS_DIR`
environment variable will need to be set to point to a clone of
the [TensorFlow Model Garden](https://github.com/tensorflow/models) repo
and the [dependencies needed to run object detection](https://github.com/tensorflow/models/blob/1efe98bb8e8d98bbffc703a90d88df15fc2ce906/research/object_detection/g3doc/installation.md#installation)
need to be installed in your environment.

The snipped below shows how to run the `1.15.2-object-detection-download-preprocess-coco-val`
container, which mounts a directory for the dataset, and runs the script
to download and preprocess the COCO validation images.
```
export DATASET_DIR=<directory where the dataset will be written>

docker run \
--env http_proxy=${http_proxy} \
--env https_proxy=${https_proxy} \
--env DATASET_DIR=${DATASET_DIR} \
-v ${DATASET_DIR}:${DATASET_DIR} \
-t amr-registry.caas.intel.com/aipg-tf/model-zoo:1.15.2-object-detection-download-preprocess-coco-val
```

After the script completes, the `DATASET_DIR` will contain a `val2017`
folder with raw images and a `tf_records` with the TF records file for
the coco validation dataset.
