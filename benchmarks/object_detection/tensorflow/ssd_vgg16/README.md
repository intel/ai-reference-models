# SSD-VGG16

This document has instructions for how to run SSD-VGG16 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference
other precisions are coming later.

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone the [original model](https://github.com/HiKapok/SSD.TensorFlow) repository:
```
$ git clone https://github.com/HiKapok/SSD.TensorFlow.git
$ cd SSD.TensorFlow
$ git checkout 2d8b0cb9b2e70281bf9dce438ff17ffa5e59075c
```

2. Clone the [intelai/models](https://github.com/intelai/models) repository.
It will be used to run the SSD-VGG16 model accuracy and inference performance tests.

3. Download the 2017 validation images file:
[COCO dataset](http://cocodataset.org/#home) and annotations:
This is required if you would like to run the accuracy test,
or batch/online inference with real data.

```
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
```

Download the validation annotations file:
```
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
```

4. Convert the COCO dataset to TF records format:

We provide a script `generate_coco_records.py` to convert the raw dataset to the TF records required pattern.
* Some dependencies are required to be installed to run the script such as `python3`, `Tensorflow` and `tqdm`, also, the `SSD.TensorFlow/dataset` from the original model directory (from step 1).

Follow the steps below get the COCO TF records:

* Copy the `generate_coco_records.py` script from `models/object_detection/tensorflow/ssd_vgg16/inference/generate_coco_records.py` 
from the `models` directory (step 2) to `SSD.TensorFlow/dataset` in the original model directory (step 1).

```
$ cp /home/<user>/models/models/object_detection/tensorflow/ssd_vgg16/inference/generate_coco_records.py /home/<user>/SSD.TensorFlow/dataset
```

* Create directory for the output TF records:
```
$ mkdir tf_records
``` 

* Run the script to generate the TF records with the required prefix `val`, COCO raw dataset and annotation file (step 3):
```
$ cd /home/<user>/SSD.TensorFlow/dataset
$ python generate_coco_records.py \
--image_path /home/<user>/val2017/ \
--annotations_file /home/<user>/annotations/instances_val2017.json \
--output_prefix val \
--output_path /home/<user>/tf_records/
```

Now, you can use the `/home/<user>/tf_records/` as the dataset location to run inference with real data, and test the model accuracy.
```
$ ls -l /home/<user>/tf_records
total 792084
-rw-r--r--. 1 <user> <group> 170038836 Mar 17 21:35 val-00000-of-00005
-rw-r--r--. 1 <user> <group> 167260232 Mar 17 21:35 val-00001-of-00005
-rw-r--r--. 1 <user> <group> 167326957 Mar 17 21:35 val-00002-of-00005
-rw-r--r--. 1 <user> <group> 166289231 Mar 17 21:35 val-00003-of-00005
-rw-r--r--. 1 <user> <group> 140168531 Mar 17 21:35 val-00004-of-00005
```

5. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssdvgg16_int8_pretrained_model.pb
```

6. Navigate to the `benchmarks` directory (step 2), and run the model scripts for either batch or online
inference or accuracy.
```
$ cd models/benchmarks
```

* Run the model for batch or online inference where the `--model-source-dir` is the model source directory from step 1,
and the `--in-graph` is the pretrained model graph from step 5.
If you specify the `--data-location` which is the path to the tf record file that you generated in step 4,
the model will run with real data, otherwise dummy data will be used:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/tf_records \
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
    * The `--data-location` is required, which is the path to the tf record file that you generated in step 4.
    * Copy the annotation file `instances_val2017.json` (from step 3) to the dataset directory `/home/<user>/tf_records/`.
    * Use the `--accuracy-only` flag:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/tf_records \
    --in-graph /home/<user>/ssdvgg16_int8_pretrained_model.pb \
    --accuracy-only \
    --batch-size 1
```

>Notes: 
>* For batch and online inference, we recommend the provided values for the arguments: `--num-inter-threads=11`, `--num-intra-threads=21`, `--data-num-inter-threads=21`,
 `--data-num-intra-threads=28` for optimized performance on `28-cores Cascade Lake (CLX)` machine.
 
>* SSD-VGG16 model accuracy test works only with the `Python3` based docker images.

>* The `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

7. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running the model for batch
and online inference, the following results are based on CLX 28-cores with hyper-threading enabled:

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

Use the steps 1, 2,3 and 4 as above.

5. Download the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssdvgg16_fp32_pretrained_model.pb
```

6. Navigate to the `benchmarks` directory (step 2), and run the model scripts for either batch
and online inference or accuracy.
```
$ cd models/benchmarks
```

* Run the model for batch and online inference where the `--model-source-dir` is the model source directory from step 1,
and the `--in-graph` is the pretrained model graph from step 5,
if you specify the `--data-location` which is the path to the tf record file that you generated in step 4,
the benchmark will run with real data, otherwise dummy data will be used:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/tf_records \
    --in-graph /home/<user>/ssdvgg16_fp32_pretrained_model.pb \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --model-name ssd_vgg16 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
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
    * Copy the annotation file `instances_val2017.json` (from step 3) to the dataset directory `/home/<user>/tf_records/`.
    * Use the `--accuracy-only` flag:
```
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir /home/<user>/SSD.TensorFlow \
    --data-location /home/<user>/tf_records \
    --in-graph /home/<user>/ssdvgg16_fp32_pretrained_model.pb \
    --accuracy-only \
    --batch-size 1
```

>Notes: 
>* For batch and online inference, we recommend the provided values for the arguments: `--num-inter-threads=11`, `--num-intra-threads=21`, `--data-num-inter-threads=21`,
 `--data-num-intra-threads=28` for optimized performance on `28-cores Cascade Lake (CLX)` machine.
 
>* SSD-VGG16 model accuracy test works only with the `Python3` based docker images.

>* The `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

7. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running batch and online inference,
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
