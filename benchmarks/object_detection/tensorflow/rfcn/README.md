# R-FCN (ResNet101)

This document has instructions for how to run R-FCN for the
following FP32 and Int8 modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for the R-FCN ResNet101 model training and inference
for other precisions are coming later.

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) as `tensorflow-models` and [intelai/models](https://github.com/intelai/models) repo and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ git clone https://github.com/IntelAI/models.git
$ cd tensorflow-models
$ git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
$ git apply ../models/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for installing dependencies and running inference as well as
converting the coco dataset to the TF records format.

2.  Download the 2017 validation
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

3. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
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

# Go back to the main models directory and get master code
$ cd /home/<user>/tensorflow-models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

4. Download the pretrained model:
  Int8 Graph
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/rfcn_resnet101_int8_coco_pretrained_model.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the scripts for either batch/online inference performance or accuracy.

```
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

Run for batch and online inference where the `--data-location`
is the path to the directory with the raw coco validation images and the
`--in-graph` is the Int8 pre-trained graph (from step 4):

```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location /home/<user>/val/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location /home/<user>/coco/output/coco_val.record \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

6. Log files are located at the value of `--output-dir` (or
`models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for batch
and online inference:
```
Step 0: 11.4450089931 seconds
Step 10: 0.25656080246 seconds
...
Step 460: 0.256786823273 seconds
Step 470: 0.267828941345 seconds
Step 480: 0.141321897507 seconds
Step 490: 0.127830982208 seconds
Avg. Duration per Step:0.195356227875
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190416_182445.log
```

And here is a sample log file tail when running for accuracy:
```
...
Accumulating evaluation results...
DONE (t=1.91s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_194752.log
```

## FP32 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for FP32 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) as `tensorflow-models` and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ cd tensorflow-models
$ git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
$ git apply models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for installing dependencies and running inference as well as
converting the coco dataset to the TF records format.

2.  Download the 2017 validation
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

3. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```
# We are going to use an older version of the conversion script to checkout the git commit
$ cd tensorflow-models
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
# Go back to the main models directory and get master code
$ cd /home/<user>/models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

4. Download the pretrained model:

  FP32 Graph
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
```


5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the scripts for either batch/online inference performance or accuracy.

```
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

Run for batch and online inference where the `--data-location`
is the path to the directory with the raw coco validation images and the
`--in-graph` is the FP32 pre-trained graph (from step 4):

```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location /home/<user>/val/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_fp32_coco_pretrained_model \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph located in the pre-trained model directory (from step 4):
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location /home/<user>/coco/output/coco_val.record \
    --in-graph /home/<user>/rfcn_resnet101_fp32_coco_pretrained_model.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

6. Log files are located at the value of `--output-dir` (or
`models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for accuracy:
```
DONE (t=1.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_rfcn_inference_fp32_20181221_211905.log
```
