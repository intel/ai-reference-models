# ResNet50

This document has instructions for how to run ResNet50 for the
following precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for ResNet50 model inference on `Int8` and `FP32`
precisions.

## Int8 Inference Instructions

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository:
```
$ git clone https://github.com/tensorflow/models.git
``` 
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

* The ImageNet dataset directory location is only required to calculate the model accuracy.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb
```

3. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

4. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

5. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance and/or calculate the accuracy.
The optimized ResNet50 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50/`.

* Calculate the model accuracy, the required parameters parameters include: the `ImageNet` dataset location (from step 1),
the pre-trained `final_int8_resnet50.pb` input graph file (from step
2, the docker image (from step 3) and the `--accuracy-only` flag.
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/dataset/FullImageNetData_directory
    --in-graph /home/myuser/resnet50_int8_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=100 \
    --accuracy-only \
    --docker-image docker_image
```
The log file is saved to the value of `--output-dir`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190104_212224.log
```

* Evaluate the model performance: The ImageNet dataset is not needed in this case:
Calculate the model throughput `images/sec`, the required parameters to run the inference script would include:
the pre-trained `resnet50_int8_pretrained_model.pb` input graph file (from step
2, the docker image (from step 3) and the `--benchmark-only` flag.

```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_int8_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=128 \
    --benchmark-only \
    --docker-image docker_image
```
The tail of the log output when the benchmarking completes should look
something like this:
```
[Running warmup steps...]
steps = 10, 460.862674539 images/sec
[Running benchmark steps...]
steps = 10, 461.002369109 images/sec
steps = 20, 460.082656541 images/sec
steps = 30, 464.707827579 images/sec
steps = 40, 463.187506632 images/sec
steps = 50, 462.725212176 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190104_213139.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..

## FP32 Inference Instructions

1. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```

2. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. If running resnet50 for accuracy, the ImageNet dataset will be
required (if running benchmarking for throughput/latency, then dummy
data will be used).

The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process, and convert the ImageNet dataset to the TF records format.

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance.
The optimized ResNet50 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50/`.
As benchmarking uses dummy data for inference, `--data-location` flag is not required.

* To measure the model latency, set `--batch-size=1` and run the benchmark script as shown:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 0.956 sec
Iteration 2: 0.018 sec
...
Iteration 38: 0.011 sec
Iteration 39: 0.011 sec
Iteration 40: 0.011 sec
Average time: 0.011 sec
Batch size = 1
Latency: 10.924 ms
Throughput: 91.541 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_215326.log
```

* To measure the model Throughput, set `--batch-size=128` and run the benchmark script as shown:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 1.777 sec
Iteration 2: 0.657 sec
Iteration 3: 0.652 sec
...
Iteration 38: 0.653 sec
Iteration 39: 0.649 sec
Iteration 40: 0.652 sec
Average time: 0.653 sec
Batch size = 128
Throughput: 196.065 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_215655.log
```

* To measure the model accuracy, use the `--accuracy-only` flag and pass
the ImageNet dataset directory from step 3 as the `--data-location`:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/myuser/dataset/ImageNetData_directory \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

The log file is saved to the value of `--output-dir`.
The tail of the log output when the accuracy run completes should look
something like this:
```
...
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7430, 0.9188)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_213452.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..