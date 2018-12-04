# SSD-MobileNet

This document has instructions for how to run SSD-MobileNet for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## FP32 Inference Instructions

1. Clone the `tensorflow/models` repository:

```
$ git clone git@github.com:tensorflow/models.git
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
      --train_image_dir="/home/myuser/coco/empty_dir" \
      --val_image_dir="/home/myuser/coco/val/val2017" \
      --test_image_dir="/home/myuser/coco/empty_dir" \
      --train_annotations_file="/home/myuser/coco/annotations/empty.json" \
      --val_annotations_file="/home/myuser/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/myuser/coco/annotations/empty.json" \
      --output_dir="/home/myuser/coco/output"

$ ll /home/myuser/coco/output
total 1598276
-rw-rw-r--. 1 myuser myuser         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 myuser myuser         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 myuser myuser 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd /home/myuser/models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

5. Download and extract the pre-trained SSD-MobileNet model from the
[TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).
The downloaded .tar file includes a `frozen_inference_graph.pb` which we
will be using when running inference.

```
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
--2018-11-02 18:21:34--  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
Resolving proxy-fm.intel.com (proxy-fm.intel.com)... 10.1.192.48
Connecting to proxy-fm.intel.com (proxy-fm.intel.com)|10.1.192.48|:911... connected.
Proxy request sent, awaiting response... 200 OK
Length: 76541073 (73M) [application/x-tar]
Saving to: ‘ssd_mobilenet_v1_coco_2018_01_28.tar.gz’

100%[==================================================================================================================================================>] 76,541,073  4.63MB/s   in 21s

2018-11-02 18:21:55 (3.53 MB/s) - ‘ssd_mobilenet_v1_coco_2018_01_28.tar.gz’ saved [76541073/76541073]

$ tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
ssd_mobilenet_v1_coco_2018_01_28/
ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.index
ssd_mobilenet_v1_coco_2018_01_28/checkpoint
ssd_mobilenet_v1_coco_2018_01_28/pipeline.config
ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.data-00000-of-00001
ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.meta
ssd_mobilenet_v1_coco_2018_01_28/saved_model/
ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb
ssd_mobilenet_v1_coco_2018_01_28/saved_model/variables/
ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb

$ cd ssd_mobilenet_v1_coco_2018_01_28

$ ll
total 58132
-rw-r--r--. 1 myuser myuser       77 Feb  1  2018 checkpoint
-rw-r--r--. 1 myuser myuser 29103956 Feb  1  2018 frozen_inference_graph.pb
-rw-r--r--. 1 myuser myuser 27380740 Feb  1  2018 model.ckpt.data-00000-of-00001
-rw-r--r--. 1 myuser myuser     8937 Feb  1  2018 model.ckpt.index
-rw-r--r--. 1 myuser myuser  3006546 Feb  1  2018 model.ckpt.meta
-rw-r--r--. 1 myuser myuser     4138 Feb  1  2018 pipeline.config
drwxr-sr-x. 3 myuser myuser     4096 Feb  1  2018 saved_model
```

6. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking, which we will
use in the next step.

```
$ git clone git@github.com:IntelAI/models.git
Cloning into 'models'...
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (7/7), done.
remote: Total 11 (delta 3), reused 4 (delta 0), pack-reused 0
Receiving objects: 100% (11/11), done.
Resolving deltas: 100% (3/3), done.
```

7. Run the `launch_benchmark.py` script from the models repo that we
just cloned, with the appropriate parameters including: the
`coco_val.record` data location (from step 4), the pre-trained
`frozen_inference_graph.pb` input graph file (from step 5, and the
location of your `tensorflow/models` clone (from step 1).

```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/coco/output/coco_val.record \
    --in-graph /home/myuser/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --single-socket \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

8. The log file is saved to:
`models/benchmarks/common/tensorflow/logs`

The tail of the log output when the benchmarking completes should look
something like this:

```
INFO:tensorflow:Processed 4970 images...
INFO:tensorflow:Processed 4980 images...
INFO:tensorflow:Processed 4990 images...
INFO:tensorflow:Processed 5000 images...
INFO:tensorflow:Finished processing records
Using model init: /workspace/benchmark/tensorflow/ssd-mobilenet/fp32/inference/model_init.py
Received these standard args: Namespace(batch_size=256, checkpoint=None, data_location='/dataset', inference_only=True, input_graph='/in_graph/frozen_inference_graph.pb', mode='inference', model_name='ssd-mobilenet', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, verbose=True)
Received these custom args: []
Initialize here.
Run model here.
current directory: /workspace/models/research
Running: OMP_NUM_THREADS=28 numactl -l -N 1 python object_detection/inference/infer_detections.py --input_tfrecord_paths /dataset --inference_graph /in_graph/frozen_inference_graph.pb --output_tfrecord_path=/tmp/ssd-mobilenet-record-out --intra_op_parallelism_threads 28 --inter_op_parallelism_threads 1 --discard_image_pixels=True --inference_only
```
