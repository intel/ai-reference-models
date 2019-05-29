# SSD-ResNet34

This document has instructions for how to run SSD-ResNet34 for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

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

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```

# We are going to use an older version of the conversion script to checkout the git commit
$ cd models
$ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40

$ cd research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"

$ ll /home/<user>/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and checkout the SHA that we are using for SSD-ResMet34
$ cd /home/<user>/models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
```

The `coco_val.record` file is what we will use in this inference example.

5. A link to download the pre-trained model is coming soon.

6. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
```

7. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-ResNet34 can be run for 
throughput and latency, or accuracy. Note that we are running
SSD-ResNet34 with a TensorFlow 1.13 docker image.

To run for throughput and latency, use the following command,
but replace in your path to the unzipped coco dataset images from step 3
for the `--dataset-location`, the path to the frozen graph that you
downloaded in step 5 as the `--in-graph`, and use the `--benchmark-only`
flag:

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_coco_pretained_model/ssd_resnet34_bs1.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image intelaipg/intel-optimized-tensorflow:1.13.1-devel-mkl-py3 \
    --benchmark-only
```

To test accuracy, use the following command but replace in your path to
the tf record file that you generated in step 4 for the `--data-location`,
the path to the frozen graph that you downloaded in step 5 as the
`--in-graph`, and use the `--accuracy-only` flag:

```
$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_coco_pretained_model/ssd_resnet34_bs1.pb \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size=1 \
    --docker-image intelaipg/intel-optimized-tensorflow:1.13.1-devel-mkl-py3 \
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
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_ssd-mobilenet_inference_fp32_20190123_225145.log
```
