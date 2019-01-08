# SSD-VGG16

This document has instructions for how to run SSD-VGG16 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference for
other precisions are coming later.

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking, the SSD-VGG16 model scripts, and instructions
for the dataset preparation.

```
$ git clone git@github.com:IntelAI/models.git
```

2. Download the VOC 2007 dataset and convert them as per
[VOC dataset](https://github.com/LevinJ/SSD_tensorflow_VOC#database) instructions.
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

3. A link to download the pre-trained model is coming soon.
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
    --precision int8 \
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
    --precision int8 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 224 \
    --socket-id 0 \
    --benchmark-only \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/ssdvgg16_int8_pretrained_model.pb
```

For latency (using `--benchmark-only` and `--batch-size 1`):
```
python launch_benchmark.py \
    --precision int8 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
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
Run batch [1/22]: 48.586711 images/sec
Run batch [2/22]: 51.355756 images/sec
Run batch [3/22]: 54.826117 images/sec
...
Run batch [19/22]: 62.367842 images/sec
Run batch [20/22]: 62.479511 images/sec
Run batch [21/22]: 62.578415 images/sec
Run batch [22/22]: 62.611104 images/sec
mAP_VOC12 = 0.645153, mAP_VOC07 = 0.632309
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20190104_234647.log
```

Example log tail when benchmarking for throughput:
```
Run warmup batch [10/10]
benchmark run
Run benchmark batch [10/100]: 55.604961 images/sec
Run benchmark batch [20/100]: 55.705138 images/sec
Run benchmark batch [30/100]: 55.789278 images/sec
Run benchmark batch [40/100]: 55.816119 images/sec
Run benchmark batch [50/100]: 55.818327 images/sec
Run benchmark batch [60/100]: 55.868167 images/sec
Run benchmark batch [70/100]: 55.842780 images/sec
Run benchmark batch [80/100]: 55.800903 images/sec
Run benchmark batch [90/100]: 55.783626 images/sec
Run benchmark batch [100/100]: 55.765641 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20190104_232204.log
```

Example log tail when benchmarking for latency:

```
benchmark run
Run benchmark batch [10/1000]: 20.041533 images/sec
Run benchmark batch [20/1000]: 20.243430 images/sec
Run benchmark batch [30/1000]: 19.927590 images/sec
...
Run benchmark batch [950/1000]: 19.920698 images/sec
Run benchmark batch [960/1000]: 19.917603 images/sec
Run benchmark batch [970/1000]: 19.920775 images/sec
Run benchmark batch [980/1000]: 19.922550 images/sec
Run benchmark batch [990/1000]: 19.922991 images/sec
Run benchmark batch [1000/1000]: 19.925384 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_int8_20190104_231640.log
```

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking, the SSD-VGG16 model scripts, and instructions
for the dataset preparation.

```
$ git clone git@github.com:IntelAI/models.git
```

2. Download the VOC 2007 dataset and convert them as per
[VOC dataset](https://github.com/LevinJ/SSD_tensorflow_VOC#database) instructions.
Update paths for `dataset_dir` and `output_dir` in `datasets/pascalvoc_to_tfrecords.py`
```
$ mkdir -p voc_dataset
$ cd voc_dataset
$ mkdir -p voc/tfrecords
$ cd voc/tfrecords
$ pwd # update this path at output_dir
$ cd ../../
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar -vxf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007
$ pwd # update this path at dataset_dir

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
-rw-r--r-- 1 kvadla kmem 176250446 Dec 17 11:26 voc_test_2007_00001-of-00003-total02000.tfrecord
-rw-r--r-- 1 kvadla kmem 177235199 Dec 17 11:26 voc_test_2007_00002-of-00003-total02000.tfrecord
-rw-r--r-- 1 kvadla kmem  84004619 Dec 17 11:26 voc_test_2007_00003-of-00003-total00952.tfrecord

$ pwd

```

3. A link to download the pre-trained model is coming soon.

4. Navigate to the `benchmarks` directory in the cloned [intelai/models](https://github.com/intelai/models) repo,
which is where the launch script is located.

```
$ cd models/benchmarks
```

Run the launch_benchmark.py script with the appropriate parameters including:
the voc tfrecords data location (from step 2) and pre-trained graph file (from step 3)

NOTE:
* `--in-graph` required for all cases. If `--benchmark-only` or `--accuracy-only` not provided, run defaults to `--benchmark-only`.
* If no `--data-location` is provided, benchmarks are based on dummy data.
* Use `--verbose` flag to see complete benchmark command args in logs.


For Latency, use `--batch-size 1`, `--socket-id 0`
```
python launch_benchmark.py \
    --precision fp32 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --in-graph /home/myuser/pre-trained/ssdvgg16_fp32_pretrained_model.pb \
    --benchmark-only
```
The tail of Latency log, looks as below.
```
...
...
Run benchmark batch [920/1000]: 25.408411 images/sec
Run benchmark batch [930/1000]: 25.409283 images/sec
Run benchmark batch [940/1000]: 25.409397 images/sec
Run benchmark batch [950/1000]: 25.407194 images/sec
Run benchmark batch [960/1000]: 25.404377 images/sec
Run benchmark batch [970/1000]: 25.400865 images/sec
Run benchmark batch [980/1000]: 25.392974 images/sec
Run benchmark batch [990/1000]: 25.392731 images/sec
Run benchmark batch [1000/1000]: 25.393458 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_fp32_20181219_234010.log

```
For Throughput, `--batch-size 224`, `--socket-id 0`
```
python launch_benchmark.py \
    --precision fp32 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 224 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --in-graph /home/myuser/pre-trained/ssdvgg16_fp32_pretrained_model.pb \
    --benchmark-only
```
The tail of Throughput log, looks as below.
```
...
...
Run benchmark batch [10/100]: 41.470731 images/sec
Run benchmark batch [20/100]: 41.507212 images/sec
Run benchmark batch [30/100]: 41.524874 images/sec
Run benchmark batch [40/100]: 41.531343 images/sec
Run benchmark batch [50/100]: 41.530304 images/sec
Run benchmark batch [60/100]: 41.525419 images/sec
Run benchmark batch [70/100]: 41.519728 images/sec
Run benchmark batch [80/100]: 41.513154 images/sec
Run benchmark batch [90/100]: 41.505310 images/sec
Run benchmark batch [100/100]: 41.500083 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_fp32_20181219_234536.log
```
For Accuracy, use `--accuracy-only`, `--batch-size 224` and `--data-location <path_to_voc_dataset>`
```
python launch_benchmark.py \
    --precision fp32 \
    --model-name ssd-vgg16 \
    --mode inference \
    --framework tensorflow \
    --batch-size 224 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --in-graph /home/myuser/pre-trained/ssdvgg16_fp32_pretrained_model.pb \
    --data-location /home/myuser/voc_dataset \
    --accuracy-only
```
The tail of Accuracy log, looks as below.
```
...
...
Run batch [15/22]: 33.434086 images/sec
Run batch [16/22]: 33.455755 images/sec
Run batch [17/22]: 33.474024 images/sec
Run batch [18/22]: 33.488637 images/sec
Run batch [19/22]: 33.496928 images/sec
Run batch [20/22]: 33.508573 images/sec
Run batch [21/22]: 33.517224 images/sec
Run batch [22/22]: 33.527867 images/sec
mAP_VOC12 = 0.651217, mAP_VOC07 = 0.637334
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 224
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_ssd-vgg16_inference_fp32_20181220_000107.log
```