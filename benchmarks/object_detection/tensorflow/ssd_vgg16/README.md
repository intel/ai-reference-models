# SSD-VGG16

This document has instructions for how to run SSD-VGG16 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other precisions are coming later.

## Int8 Inference Instructions

1. Clone the [original model](https://github.com/HiKapok/SSD.TensorFlow) repository:
```
$ git clone https://github.com/HiKapok/SSD.TensorFlow.git
$ cd SSD.TensorFlow
$ git checkout 2d8b0cb9b2e70281bf9dce438ff17ffa5e59075c
```

2. Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:
This is required if you would like to run the accuracy test,
or the throughput and latency benchmark with real data.

The [TensorFlow models](https://github.com/tensorflow/models) repo will be used for
converting the coco dataset to the TF records format.
Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
```
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd ..
```

Continue the instructions below to generate the
TF record file.
```
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
TF records format in order to use it with the inference script. We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed. 

```
# We are going to use an older version of the conversion script to checkout the git commit
$ git clone https://github.com/tensorflow/models.git
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

$ ll /home/myuser/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record
```

4. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssdvgg16_int8_pretrained_model.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the benchmarking scripts for either benchmarking throughput
and latency or accuracy.
```
$ git clone git@github.com:IntelAI/models.git
$ cd benchmarks
```

* Run benchmarking for throughput and latency where the `--model-source-dir` is the model source directory from step 1,
and the `--in-graph` is the pretrained model graph from step 4,
if you specify the `--data-location` which is the path to the tf record file that you generated in step 3,
the benchmark will run with real data, otherwise dummy data will be used:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:nightly-master-devel-mkl-py3 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/coco/output \
    --in-graph /home/<user>/ssdvgg16_int8_pretrained_model.pb \
    --batch-size 1 \
    --socket-id 0 \
    --num-inter-threads 11 \
    --num-intra-threads 21 \
    --data-num-inter-threads 21 \
    --data-num-intra-threads 28 \
    -- warmup-steps=100 steps=500
```

* For the accuracy test:

    * Clone the customized [cocoapi repo](https://github.com/waleedka/coco) in
the model directory `SSD.TensorFlow` from step 1.
    ```
    $ git clone https://github.com/waleedka/coco.git

    ```
    * The `--data-location` is required, which is the path to the tf record file that you generated in step 3.
    * Copy the annotation file `instances_val2017.json` (from step 3) to the dataset directory `/home/<user>/coco/output`.
    * Use the `--accuracy-only` flag:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:nightly-master-devel-mkl-py3 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/coco/output \
    --in-graph /home/<user>/ssdvgg16_int8_pretrained_model.pb \
    --accuracy-only \
    --batch-size 1
```

>Notes: 
>* For the throughput and latency benchmark, we recommend the provided values for the arguments: `--num-inter-threads=11`, `--num-intra-threads=21`, `--data-num-inter-threads=21`,
 `--data-num-intra-threads=28` for optimized performance on `28-cores Cascade Lake (CLX)` machine.
 
>* SSD-VGG16 model accuracy test works only with the `Python3` based docker images.

>* The `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

6. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running benchmarking for throughput
and latency, the following results are based on CLX 28-cores with hyper-threading enabled:

```
Batch size = 1
Throughput: 30.382 images/sec
Latency: 32.915 ms
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_ssd_vgg16_inference_int8_20190417_231832.log
```

And here is a sample log file tail when running for accuracy:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```

## FP32 Inference Instructions

1. Clone the [original model](https://github.com/HiKapok/SSD.TensorFlow) repository:
```
$ git clone https://github.com/HiKapok/SSD.TensorFlow.git
$ cd SSD.TensorFlow
$ git checkout 2d8b0cb9b2e70281bf9dce438ff17ffa5e59075c
```

2. Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

This is required if you would like to run the accuracy test,
or the throughput and latency benchmark with real data.

The [TensorFlow models](https://github.com/tensorflow/models) repo will be used for
converting the coco dataset to the TF records format.
Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
```
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd ..
```

Continue the instructions below to generate the
TF record file.
```
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
TF records format in order to use it with the inference script. We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed. 

```
# We are going to use an older version of the conversion script to checkout the git commit
$ git clone https://github.com/tensorflow/models.git
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

$ ll /home/myuser/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record
```

4. Download the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssdvgg16_fp32_pretrained_model.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the benchmarking scripts for either benchmarking throughput
and latency or accuracy.
```
$ git clone git@github.com:IntelAI/models.git
$ cd benchmarks
```

* Run benchmarking for throughput and latency where the `--model-source-dir` is the model source directory from step 1,
and the `--in-graph` is the pretrained model graph from step 4,
if you specify the `--data-location` which is the path to the tf record file that you generated in step 3,
the benchmark will run with real data, otherwise dummy data will be used:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output \
    --in-graph /home/<user>/ssdvgg16_fp32_pretrained_model.pb \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --model-name ssd_vgg16 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intelaipg/intel-optimized-tensorflow:nightly-master-devel-mkl-py3 \
    --batch-size 1 \
    --socket-id 0 \
    --num-inter-threads 11 \
    --num-intra-threads 21 \
    --data-num-inter-threads 21 \
    --data-num-intra-threads 28 \
    -- warmup-steps=100 steps=500
```

* For the accuracy test:

    * Clone the customized [cocoapi repo](https://github.com/waleedka/coco) in
the model directory `SSD.TensorFlow` from step 1.
    ```
    $ git clone https://github.com/waleedka/coco.git

    ```
    * The `--data-location` is required, which is the path to the tf record file that you generated in step 3.
    * Copy the annotation file `instances_val2017.json` (from step 3) to the dataset directory `/home/<user>/coco/output`.
    * Use the `--accuracy-only` flag:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:nightly-master-devel-mkl-py3 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/coco/output \
    --in-graph /home/<user>/ssdvgg16_fp32_pretrained_model.pb \
    --accuracy-only \
    --batch-size 1
```

>Notes: 
>* For the throughput and latency benchmark, we recommend the provided values for the arguments: `--num-inter-threads=11`, `--num-intra-threads=21`, `--data-num-inter-threads=21`,
 `--data-num-intra-threads=28` for optimized performance on `28-cores Cascade Lake (CLX)` machine.
 
>* SSD-VGG16 model accuracy test works only with the `Python3` based docker images.

>* The `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

6. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running throughput and latency benchmarking,
the following results are based on CLX 28-cores with hyper-threading enabled:

```
Batch size = 1
Throughput: 15.662 images/sec
Latency: 63.848 ms
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_ssd_vgg16_inference_fp32_20190417_232130.log
```

Below is a sample log file tail when testing accuracy:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
```
