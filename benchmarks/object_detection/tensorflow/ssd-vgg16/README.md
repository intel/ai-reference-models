# SSD-VGG16

This document has instructions for how to run SSD-VGG16 for the
following modes/platforms:
* [Int8 inference](#int8-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking, the SSD-VGG16 model scripts, and instructions
for the dataset preparation. 

```
$ git clone git@github.com:IntelAI/models.git
```

2. Download the VOC 2007 dataset and convert them as per
VOC dataset instructions at `/home/myuser/intelai/models/models/object_detection/tensorflow/ssd-vgg16/README.md#database`.
Update paths for `dataset_dir` and `output_dir` in `datasets/pascalvoc_to_tfrecords.py`
```
$ mkdir -p voc_dataset
$ cd voc_dataset
$ mkdir -p tfrecords
$ cd tfrecords
$ pwd # update this path at output_dir
$ cd ..
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar -vxf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007
$ pwd # update this path at dataset_dir
$ cd ../../../

# Setup to convert dataset to TFrecords. Use `virtualenv` to avoid conflicts

$ sudo apt-get install python-pip
$ pip install virtualenv
$ virtualenv venv

$ source venv/bin/activate
$ pip install tensorflow

# Add SSD_tensorflow_VOC repo path to PYTHONPATH
$ export PYTHONPATH=<path_to_SSD_tensorflow_VOC>:$PYTHONPATH

# cd SSD_tensorflow_VOC repo
$ SSD_tensorflow_VOC/datasets
$ python pascalvoc_to_tfrecords.py
...
...
Converting image 4944/4952 009948 shard 3
Converting image 4945/4952 009951 shard 3
Converting image 4946/4952 009952 shard 3
Converting image 4947/4952 009953 shard 3
Converting image 4948/4952 009956 shard 3
Converting image 4949/4952 009957 shard 3
Converting image 4950/4952 009960 shard 3
Converting image 4951/4952 009962 shard 3
Converting image 4952/4952 009963 shard 3

Finished converting the Pascal VOC dataset!

$ cd path_to_tfrecords> ls -lrt
total 427252
-rw-r--r-- 1 user user 176250446 Dec 17 11:26 voc_test_2007_00001-of-00003-total02000.tfrecord
-rw-r--r-- 1 user user 177235199 Dec 17 11:26 voc_test_2007_00002-of-00003-total02000.tfrecord
-rw-r--r-- 1 user user  84004619 Dec 17 11:26 voc_test_2007_00003-of-00003-total00952.tfrecord

$ pwd

```

3. Download the pre-trained model
```
$ mkdir -p pre-trained && cd pre-trained
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow-staging/models/ssdvgg16_int8_pretrained_model.pb
$ pwd

```
4. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).


5. Navigate to the `benchmarks` directory in the cloned [intelai/models](https://github.com/intelai/models) repo,
which is where the launch script is located.

```
$ cd models/benchmarks
```

Run the launch_benchmark.py script with the appropriate parameters including:
the voc tfrecords data location (from step 2),
the pre-trained graph file (from step 3), and the docker image (from step 4).

SSD-VGG16 can be run to test accuracy or benchmarking throughput or
latency. Use one of the following examples below, depending on your use
case.

For accuracy (using your `--accuracy-only`, `--data-location` and
`--batch-size 224`):

```
python launch_benchmark.py \
    --platform int8 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 224 \
    --accuracy-only \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/ssdvgg16_int8_pretrained_model.pb \
    --data-location /home/myuser/tmp/voc_dataset
```

For throughput benchmarking (using `--benchmark-only` and `--batch-size 224`):
```
python launch_benchmark.py \
    --platform int8 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 224 \
    --single-socket \
    --benchmark-only \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=ssd-vgg16 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=224  --accuracy-only  --verbose --in-graph=/in_graph/ssdvgg16_int8_pretrained_model.pb --data-location=/dataset --in-graph=/in_graph/final_intel_qmodel_ssd.pb     --data-location=/dataset
```

For latency (using `--benchmark-only` and `--batch-size 1`):
```
python launch_benchmark.py \
    --platform int8 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --single-socket \
    --benchmark-only \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/ssdvgg16_int8_pretrained_model.pb
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

6. The log file is saved to the
`intelai/models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Run batch [1/22]: 48.562328 images/sec
Run batch [2/22]: 50.122229 images/sec
Run batch [3/22]: 52.681343 images/sec
...
Run batch [11/22]: 60.480486 images/sec
Run batch [12/22]: 60.945945 images/sec
Run batch [13/22]: 61.259607 images/sec
Run batch [14/22]: 61.518968 images/sec
Run batch [15/22]: 61.706550 images/sec
Run batch [16/22]: 61.931511 images/sec
...
Run batch [22/22]: 62.889507 images/sec
mAP_VOC12 = 0.645188, mAP_VOC07 = 0.632397
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20181220_210049.log
```

Example log tail when benchmarking for throughput:
```
Run warmup batch [10/10]
benchmark run
Run benchmark batch [10/100]: 57.222400 images/sec
Run benchmark batch [20/100]: 57.156093 images/sec
Run benchmark batch [30/100]: 57.162007 images/sec
Run benchmark batch [40/100]: 57.177111 images/sec
Run benchmark batch [50/100]: 57.168034 images/sec
Run benchmark batch [60/100]: 57.146903 images/sec
Run benchmark batch [70/100]: 57.133651 images/sec
Run benchmark batch [80/100]: 57.120339 images/sec
Run benchmark batch [90/100]: 57.111009 images/sec
Run benchmark batch [100/100]: 57.110389 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20181220_200808.log
```

Example log tail when benchmarking for latency:

```
benchmark run
Run benchmark batch [10/1000]: 19.677674 images/sec
Run benchmark batch [20/1000]: 19.667542 images/sec
...
Run benchmark batch [210/1000]: 19.961509 images/sec
Run benchmark batch [220/1000]: 19.982158 images/sec
Run benchmark batch [230/1000]: 20.005125 images/sec
Run benchmark batch [240/1000]: 20.030552 images/sec
...
Run benchmark batch [990/1000]: 20.145094 images/sec
Run benchmark batch [1000/1000]: 20.140605 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20181220_202603.log
```