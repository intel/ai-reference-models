# SSD-ResNet34

This document has instructions for how to run SSD-ResNet34 for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)
* [INT8 inference](#int8-inference-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Clone the `tensorflow/models` repository with the specified SHA,
since we are using an older version of the models repo for
SSD-ResNet34.

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

2. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

3.  Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

```
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd ..

$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ cd ..
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd ..
```

4. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```
# start the data format convert
$ cd models/research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"
      
# It will generate several tfrecord files in the directory "/home/<user>/coco/output", such as:
coco_testdev.record-000[00-99]-of-00100
coco_train.record-000[00-99]-of-00100
coco_val.record-0000[0-9]-of-00010

# Rename the validation datasets from coco_val.record-0000[0-9]-of-00010 to validation-0000[0-9]-of-00010

# Copy the annotations directory into the "/home/<user>/coco/output"
$ ll /home/<user>/coco/output/annotations
captions_train2017.json  instances_train2017.json  person_keypoints_train2017.json
captions_val2017.json    instances_val2017.json    person_keypoints_val2017.json
```

The `validation-0000[0-9]-of-00010` file is what we will use in this inference example.

5. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd_resnet34_fp32_bs1_pretrained_model.pb
```

6. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
```

7. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-ResNet34 can be run for 
batch and online inference, or accuracy. Note that we are running
SSD-ResNet34 with a TensorFlow 1.14 docker image.

To run for batch and online inference, use the following command,
the path to the frozen graph that you downloaded in step 5 as 
the `--in-graph`, and use the `--benchmark-only`
flag:

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/ssd_resnet34_fp32_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --benchmark-only
```

To test accuracy, use the following command but replace in your path to
the tf record file that you generated in step 4 for the `--data-location`,
the path to the frozen graph that you downloaded in step 5 as the
`--in-graph`, and use the `--accuracy-only` flag:

```
$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_fp32_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --accuracy-only
```

8. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for performance:

```
Batchsize: 1
Time spent per BATCH:    21.8225 ms
Total samples/sec:    45.8243 samples/s
```

Below is a sample log file tail when testing accuracy:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494
Current AP: 0.21082
```

## INT8 Inference Instructions

1. Clone the `tensorflow/models` repository with the specified SHA,
since we are using an older version of the models repo for
SSD-ResNet34.

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

2. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

3.  Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

```
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd ..

$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ cd ..
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd ..
```

4. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```
# start the data format convert
$ cd models/research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"
      
# It will generate several tfrecord files in the directory "/home/<user>/coco/output", such as:
coco_testdev.record-000[00-99]-of-00100
coco_train.record-000[00-99]-of-00100
coco_val.record-0000[0-9]-of-00010

# Rename the validation datasets from coco_val.record-0000[0-9]-of-00010 to validation-0000[0-9]-of-00010

# Copy the annotations directory into the "/home/<user>/coco/output"
$ ll /home/<user>/coco/output/annotations
captions_train2017.json  instances_train2017.json  person_keypoints_train2017.json
captions_val2017.json    instances_val2017.json    person_keypoints_val2017.json
```

The `validation-0000[0-9]-of-00010` file is what we will use in this inference example.

5. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd_resnet34_int8_bs1_pretrained_model.pb
```

6. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
```

7. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-ResNet34 can be run for testing batch or online inference, or testing accuracy. Note that we are running
SSD-ResNet34 with a TensorFlow 1.14 docker image.

To run for batch and online inference, use the following command,
the path to the frozen graph that you downloaded in step 5 as
the `--in-graph`, and use the `--benchmark-only`
flag:

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/ssd_resnet34_int8_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --benchmark-only
```

To test accuracy, use the following command but replace in your path to
the tf record file that you generated in step 4 for the `--data-location`,
the path to the frozen graph that you downloaded in step 5 as the
`--in-graph`, and use the `--accuracy-only` flag:

```
$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_int8_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --accuracy-only
```

8. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when testing performance:

```
Batchsize: 1
Time spent per BATCH:    12.0245 ms
Total samples/sec:    83.1635 samples/s
```

Below is a sample log file tail when testing accuracy:

```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.208
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.051
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
Current AP: 0.20408
```
