# SSD-ResNet34

This document has instructions for how to run SSD-ResNet34 for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)
* [INT8 inference](#int8-inference-instructions)
* [FP32 Training](#fp32-training-instructions)
* [BF16 Training](#bf16-training-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Please ensure you have installed all the libraries listed in the 
`requirements` before you start the next step.

2. Clone the `tensorflow/models` repository with the specified SHA,
since we are using an older version of the models repo for
SSD-ResNet34.

```
$ git clone https://github.com/tensorflow/models.git tf_models
$ cd tf_models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

3. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

4.  Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

```
cd $MODEL_WORK_DIR
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd $MODEL_WORK_DIR

$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ cd $MODEL_WORK_DIR
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd $MODEL_WORK_DIR
```

5. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 3.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```

# We are going to use an older version of the conversion script to checkout the git commit
$ cd tf_models
$ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40

$ cd research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --val_image_dir="$MODEL_WORK_DIR/val/val2017" \
      --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --train_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --val_annotations_file="$MODEL_WORK_DIR/annotations/annotations/instances_val2017.json" \
      --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --output_dir="$MODEL_WORK_DIR/output"

$ ll $MODEL_WORK_DIR/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and checkout the SHA that we are using for SSD-ResNet34
$ cd /home/<user>/tf_models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
```

The `coco_val.record` file is what we will use in this inference example.
```
$ mv /home/<user>/coco/output/coco_val.record /home/<user>/coco/output/validation-00000-of-00001
```

6. Download the pretrained model:

```
# ssd-resnet34 300x300
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_fp32_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200 
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
```

7. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
```

8. Clone the [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks.git) repo. This repo contains the method needed
to run the ssd-resnet34 model. Please ensure that the `ssd-resnet-benchmarks` and `models` repos cloned in step 7 and step 8 
are in the same folder.

```
$ git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
$ cd ssd-resnet-benchmarks
$ git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
$ cd ../
```

9. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-ResNet34 can be run for 
batch and online inference, or accuracy. Note that we are running
SSD-ResNet34 with a TensorFlow 2.1.0 docker image.

To run for batch and online inference, use the following command,
the path to the frozen graph that you downloaded in step 5 as 
the `--in-graph`, and use the `--benchmark-only` flag. If you run on docker mode, you also need to provide `ssd-resnet-benchmarks` path for `volume` flag.
By default it runs with input size 300x300, you may add `-- input-size=1200` 
flag to run benchmark with input size 1200x1200.

```
$ cd $MODEL_WORK_DIR/models/benchmarks

# benchmarks with input size 1200x1200
$ python launch_benchmark.py \
    --in-graph /home/<user>/ssd_resnet34_fp32_1200x1200_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --benchmark-only \
    -- input-size=1200
```

To run the accuracy test, use the following command but replace in your path to
the tf record file that you generated in step 5 for the `--data-location`,
the path to the frozen graph that you downloaded in step 6 as the
`--in-graph`, and use the `--accuracy-only` flag. By default it runs with 
input size 300x300, you may add `-- input-size=1200` flag last to run the test with 
input size 1200x1200.

```
# accuracy test with input size 300x300
$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_fp32_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --accuracy-only 
```

10. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for performance:

```
Batchsize: 1
Time spent per BATCH:    ... ms
Total samples/sec:    ... samples/s
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

10. To return to where you started from:
```
$ popd
```

## INT8 Inference Instructions
1. Please ensure you have installed all the libraries listed in the 
`requirements` before you start the next step.

2. Clone the `tensorflow/models` repository with the specified SHA,
since we are using an older version of the models repo for
SSD-ResNet34.

```
$ git clone https://github.com/tensorflow/models.git tf_models
$ cd tf_models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

3. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

4.  Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

```
cd $MODEL_WORK_DIR
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip

$ cd $MODEL_WORK_DIR
$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ cd $MODEL_WORK_DIR
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd $MODEL_WORK_DIR
```

5. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```

# We are going to use an older version of the conversion script to checkout the git commit
$ cd tf_models
$ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40

$ cd research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --val_image_dir="$MODEL_WORK_DIR/val/val2017" \
      --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --train_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --val_annotations_file="$MODEL_WORK_DIR/annotations/annotations/instances_val2017.json" \
      --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --output_dir="$MODEL_WORK_DIR/output"

$ ll $MODEL_WORK_DIR/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and checkout the SHA that we are using for SSD-ResNet34
$ cd /home/<user>/tf_models
$ git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
```

The `coco_val.record` file is what we will use in this inference example.
```
$ mv /home/<user>/coco/output/coco_val.record /home/<user>/coco/output/validation-00000-of-00001
```

6. Download the pretrained model:

```
# ssd-resnet34 300x300
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_int8_bs1_pretrained_model.pb
# ssd-resnet34 1200x1200 
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```

If want to download the pretrained model for `--input-size=1200`, use the command below instead.
```

```

7. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
```

8. Clone the [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks.git) repo. This repo contains the method needed
to run the ssd-resnet34 model. Please ensure that the `ssd-resnet-benchmarks` and `models` repos cloned in step 7 and step 8 
are in the same folder.

```
$ git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
$ cd ssd-resnet-benchmarks
$ git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
$ cd ../
```

9. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-ResNet34 can be run for testing batch or online inference, or testing accuracy. Note that we are running
SSD-ResNet34 with a TensorFlow 2.1.0 docker image.

To run for batch and online inference, use the following command,
the path to the frozen graph that you downloaded in step 5 as 
the `--in-graph`, and use the `--benchmark-only` flag. If you run on docker mode, you also need to provide `ssd-resnet-benchmarks` path for `volume` flag.
By default it runs with input size 300x300, you may add `-- input-size=1200` 
flag to run benchmark with input size 1200x1200.

```
$ cd $MODEL_WORK_DIR/models/benchmarks

# benchmarks with input size 300x300
$ python launch_benchmark.py \
    --in-graph /home/<user>/ssd_resnet34_int8_bs1_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --benchmark-only 
```

To run the accuracy test, use the following command but replace in your path to
the tf record file that you generated in step 5 for the `--data-location`,
the path to the frozen graph that you downloaded in step 6 as the
`--in-graph`, and use the `--accuracy-only` flag. By default it runs with 
input size 300x300, you may add `-- input-size=1200` flag to run the test with 
input size 1200x1200.

```
# accuracy test with input size 1200x1200
$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --in-graph /home/<user>/ssd_resnet34_int8_1200x1200_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --accuracy-only \
    -- input-size=1200
```

10. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when testing performance:

```
Batchsize: 1
Time spent per BATCH:    ... ms
Total samples/sec:    ... samples/s
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

10. To return to where you started from:
```
$ popd
```

## FP32 Training Instructions

1. Store the path to the current directory:
```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR
```

2. Clone the [intelai/models](https://github.com/intelai/models) repository. This repository has the launch script for running the model.

   ```bash
   $ cd $MODEL_WORK_DIR
   $ git clone https://github.com/IntelAI/models.git
   ```

3. Clone the `tensorflow/models` repository with the specified SHA, since we are using an older version of the models repository for SSD-ResNet34.

   ```bash
   $ git clone https://github.com/tensorflow/models.git tf_models
   $ cd tf_models
   $ git checkout 8110bb64ca63c48d0caee9d565e5b4274db2220a
   $ git apply $MODEL_WORK_DIR/models/models/object_detection/tensorflow/ssd-resnet34/training/fp32/tf-2.0.diff
   ```

   The TensorFlow models repository will be used for running training as well as converting the coco dataset to the TF records format.

4. Follow the TensorFlow models object detection [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation) to get your environment setup with the required dependencies.

5. Download the 2017 train [COCO dataset](http://cocodataset.org/#home):

   ```bash
   $ cd $MODEL_WORK_DIR
   $ mkdir train
   $ cd train
   $ wget http://images.cocodataset.org/zips/train2017.zip
   $ unzip train2017.zip
   
   $ cd $MODEL_WORK_DIR
   $ mkdir val
   $ cd val
   $ wget http://images.cocodataset.org/zips/val2017.zip
   $ unzip val2017.zip
   
   $ cd $MODEL_WORK_DIR
   $ mkdir annotations
   $ cd annotations
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   $ unzip annotations_trainval2017.zip
   ```

   Since we are only using the train and validation dataset in this example, we will create an empty directory and empty annotations json file to pass as the test directories in the next step.

   ```
   $ cd $MODEL_WORK_DIR
   $ mkdir empty_dir
   
   $ cd annotations
   $ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
   $ cd $MODEL_WORK_DIR
   ```

6. Now that you have the raw COCO dataset, we need to convert it to the TF records format in order to use it with the training script. We will do this by running the `create_coco_tf_record.py` file in the TensorFlow models repository.

   ```bash
   # We are going to use an older version of the conversion script to checkout the git commit
   $ cd models
   $ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40
   
   $ cd research/object_detection/dataset_tools/
   $ python create_coco_tf_record.py --logtostderr \
         --train_image_dir="$MODEL_WORK_DIR/train2017" \
         --val_image_dir="$MODEL_WORK_DIR/val2017" \
         --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
         --train_annotations_file="$MODEL_WORK_DIR/annotations/instances_train2017.json" \
         --val_annotations_file="$MODEL_WORK_DIR/annotations/instances_val2017.json" \
         --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
         --output_dir="$MODEL_WORK_DIR/output"
   ```
   
   The `coco_train.record-*-of-*` files are what we will use in this training example.


7. Next, navigate to the `benchmarks` directory of the [intelai/models](https://github.com/intelai/models) repository that was just cloned in the previous step.

   To run for training, use the following command.

   ```bash
   $ cd $MODEL_WORK_DIR/models/benchmarks/
   
    $ python3 launch_benchmark.py \
    --data-location /path/to/coco-dataset \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name ssd-resnet34 --framework tensorflow \
    --precision fp32 --mode training \
    --num-train-steps 100 --num-cores 52 \
    --num-inter-threads 1 --num-intra-threads 52 \
    --batch-size=52 --weight_decay=1e-4 \
    --mpi_num_processes=1 --mpi_num_processes_per_socket=1 \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
   ```

## BF16 Training Instructions
(Experimental)

   1. Follow steps 1-7 from the above FP32 Training Instructions to setup the environment.

   2. Next, navigate to the benchmarks directory of the intelai/models repository that was cloned earlier.
      Use the below command to run the model in benchmarking mode:

      Note: for best performance, use the same value for the arguments num-cores and num-intra-thread as follows:
        For single instance run (mpi_num_processes=1): the value is equal to number of logical cores per socket.
        For multi-instance run (mpi_num_processes > 1): the value is equal to (#_of_logical_cores_per_socket - 2).

      ```bash
      $ cd $MODEL_WORK_DIR/models/benchmarks/

      $ python3 launch_benchmark.py \
      --data-location <path to coco_training_dataset> \
      --model-source-dir <path to tf_models> \
      --model-name ssd-resnet34 --framework tensorflow \
      --precision bfloat16 --mode training \
      --num-train-steps 100 --num-cores 52 \
      --num-inter-threads 1 --num-intra-threads 52 \
      --batch-size=100 --weight_decay=1e-4 \
      --mpi_num_processes=1 --mpi_num_processes_per_socket=1 \
      --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly

      ```

   3. To run training and achieve convergence, download the backbone model from the links below , then use the folowing command:

      https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/checkpoint

      https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.data-00000-of-00001

      https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.index

      https://storage.googleapis.com/tf-perf-public/resnet34_ssd_checkpoint/model.ckpt-28152.meta

      Place the above files in one directory, and pass that location below as --backbone-model.

      ```bash

      $ python3 launch_benchmark.py \
      --data-location <path to coco_training_dataset> \
      --model-source-dir <path to tf_models> \
      --model-name ssd-resnet34 --framework tensorflow \
      --precision bfloat16 --mode training \
      --num-cores 50 --num-inter-threads 1 \
      --num-intra-threads 50 --batch-size=100 --mpi_num_processes=4 \
      --mpi_num_processes_per_socket=1 \
      --checkpoint <path to output_train_directory> \
      --backbone-model <path to resnet34_backbone_trained_model> \
      --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly

      ```
   
   4. To run in eval mode (to check accuracy) if checkpoints are available. Use the below command:

      Note that --data-location now points to the location of COCO validation dataset.

      ```bash

      $ python3 launch_benchmark.py \
      --data-location <path to coco_validation_dataset> \
      --model-source-dir <path to tf_models> \
      --model-name ssd-resnet34 --framework tensorflow \
      --precision bfloat16 --mode training \
      --num-cores 52 --num-inter-threads 1 \
      --num-intra-threads 52 --batch-size=100 --mpi_num_processes=1 \
      --mpi_num_processes_per_socket=1 --accuracy-only \
      --checkpoint <path to pretrained_checkpoints> \
      --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly

      ```
