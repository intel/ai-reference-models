# ResNet50

This document has instructions for how to run ResNet50 for the
following platforms:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for ResNet50 model inference on `Int8` and `FP32`
platforms.

## Int8 Inference Instructions

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository:
```
$ git clone git@github.com:tensorflow/models.git
``` 
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

* The ImageNet dataset directory location is only required to calculate the model accuracy.

2. Download the pre-trained ResNet50 model:

```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.tar.gz
$ tar -xzvf resnet50_int8_pretrained_model.tar.gz 
resnet50_int8_pretrained_model/
resnet50_int8_pretrained_model/final_int8_resnet50.pb
```

3. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

4. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone git@github.com:IntelAI/models.git
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
    --in-graph /home/myuser/resnet50_int8_pretrained_model/final_int8_resnet50.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --platform int8 \
    --mode inference \
    --batch-size=100 \
    --accuracy-only \
    --docker-image docker_image
```
The log file is saved to:
`models/benchmarks/common/tensorflow/logs`.

The tail of the log output when the benchmarking completes should look
something like this:
```
(Top1 accuracy, Top5 accuracy) = (0.7346, 0.9144)
Processed 44600 images. (Top1 accuracy, Top5 accuracy) = (0.7346, 0.9143)
Processed 44700 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9144)
Processed 44800 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9143)
Processed 44900 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9143)
Processed 45000 images. (Top1 accuracy, Top5 accuracy) = (0.7344, 0.9144)
Processed 45100 images. (Top1 accuracy, Top5 accuracy) = (0.7344, 0.9144)
Processed 45200 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9145)
Processed 45300 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9145)
Processed 45400 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9145)
Processed 45500 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9145)
Processed 45600 images. (Top1 accuracy, Top5 accuracy) = (0.7346, 0.9145)
Processed 45700 images. (Top1 accuracy, Top5 accuracy) = (0.7347, 0.9145)
Processed 45800 images. (Top1 accuracy, Top5 accuracy) = (0.7345, 0.9144)
Processed 45900 images. (Top1 accuracy, Top5 accuracy) = (0.7346, 0.9144)
Processed 46000 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9145)
Processed 46100 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9146)
Processed 46200 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9145)
Processed 46300 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9145)
Processed 46400 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9146)
Processed 46500 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9146)
Processed 46600 images. (Top1 accuracy, Top5 accuracy) = (0.7347, 0.9146)
Processed 46700 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9146)
Processed 46800 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9146)
Processed 46900 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9146)
Processed 47000 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9147)
Processed 47100 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9146)
Processed 47200 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9146)
Processed 47300 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9145)
Processed 47400 images. (Top1 accuracy, Top5 accuracy) = (0.7346, 0.9144)
Processed 47500 images. (Top1 accuracy, Top5 accuracy) = (0.7347, 0.9144)
Processed 47600 images. (Top1 accuracy, Top5 accuracy) = (0.7348, 0.9145)
Processed 47700 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9145)
Processed 47800 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9145)
Processed 47900 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9144)
Processed 48000 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9144)
Processed 48100 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9144)
Processed 48200 images. (Top1 accuracy, Top5 accuracy) = (0.7349, 0.9145)
Processed 48300 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9145)
Processed 48400 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9145)
Processed 48500 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9144)
Processed 48600 images. (Top1 accuracy, Top5 accuracy) = (0.7350, 0.9145)
Processed 48700 images. (Top1 accuracy, Top5 accuracy) = (0.7353, 0.9146)
Processed 48800 images. (Top1 accuracy, Top5 accuracy) = (0.7353, 0.9146)
Processed 48900 images. (Top1 accuracy, Top5 accuracy) = (0.7354, 0.9147)
Processed 49000 images. (Top1 accuracy, Top5 accuracy) = (0.7356, 0.9148)
Processed 49100 images. (Top1 accuracy, Top5 accuracy) = (0.7356, 0.9149)
Processed 49200 images. (Top1 accuracy, Top5 accuracy) = (0.7357, 0.9148)
Processed 49300 images. (Top1 accuracy, Top5 accuracy) = (0.7356, 0.9148)
Processed 49400 images. (Top1 accuracy, Top5 accuracy) = (0.7358, 0.9149)
Processed 49500 images. (Top1 accuracy, Top5 accuracy) = (0.7358, 0.9149)
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7359, 0.9148)
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7359, 0.9149)
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7359, 0.9149)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7359, 0.9149)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9149)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/final_int8_resnet50.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100 --data-location=/dataset --single-socket --verbose --accuracy-only                          --in-graph=/in_graph/final_int8_resnet50.pb
Batch Size: 100
Ran inference with batch size 100
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log
```

* Evaluate the model performance: The ImageNet dataset is not needed in this case:
Calculate the model throughput `images/sec`, the required parameters to run the inference script would include:
the pre-trained `final_int8_resnet50.pb` input graph file (from step
2, the docker image (from step 3) and the `--benchmark-only` flag.

```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_int8_pretrained_model/final_int8_resnet50.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --platform int8 \
    --mode inference \
    --batch-size=128 \
    --benchmark-only \
    --docker-image docker_image
```
The tail of the log output when the benchmarking completes should look
something like this:
```
steps = 10, 462.431070436 images/sec
[Running benchmark steps...]
steps = 10, 465.158375557 images/sec
steps = 20, 469.24763528 images/sec
steps = 30, 467.694254776 images/sec
steps = 40, 470.733760164 images/sec
steps = 50, 468.407939199 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/final_int8_resnet50.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --data-location=/dataset --single-socket --verbose              --benchmark-only             --in-graph=/in_graph/final_int8_resnet50.pb
Batch Size: 128
Ran inference with batch size 128
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log
```

## FP32 Inference Instructions

1. Download the pre-trained ResNet50 model:

```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.tar.gz
$ tar -xzvf resnet50_fp32_pretrained_model.tar.gz 
resnet50_fp32_pretrained_model/
resnet50_fp32_pretrained_model/freezed_resnet50.pb
```
2. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone git@github.com:IntelAI/models.git
```

3. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance.
The optimized ResNet50 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50/`.
As benchmarking uses dummy data for inference, `--data-location` flag is not required.

* To measure the model latency, set `--batch-size=1` and run the benchmark script as shown:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_fp32_pretrained_model/freezed_resnet50.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --batch-size=1 \
    --single-socket \
    --docker-image docker_image \
    --verbose
```

The log file is saved to:
`models/benchmarks/common/tensorflow/logs`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 0.978 sec
Iteration 2: 0.011 sec
Iteration 3: 0.011 sec
Iteration 4: 0.012 sec
Iteration 5: 0.011 sec
Iteration 6: 0.011 sec
Iteration 7: 0.011 sec
Iteration 8: 0.011 sec
Iteration 9: 0.011 sec
Iteration 10: 0.011 sec
Iteration 11: 0.011 sec
Iteration 12: 0.011 sec
Iteration 13: 0.011 sec
Iteration 14: 0.011 sec
Iteration 15: 0.011 sec
Iteration 16: 0.011 sec
Iteration 17: 0.011 sec
Iteration 18: 0.011 sec
Iteration 19: 0.011 sec
Iteration 20: 0.011 sec
Iteration 21: 0.011 sec
Iteration 22: 0.011 sec
Iteration 23: 0.011 sec
Iteration 24: 0.011 sec
Iteration 25: 0.011 sec
Iteration 26: 0.011 sec
Iteration 27: 0.011 sec
Iteration 28: 0.011 sec
Iteration 29: 0.011 sec
Iteration 30: 0.011 sec
Iteration 31: 0.011 sec
Iteration 32: 0.011 sec
Iteration 33: 0.011 sec
Iteration 34: 0.011 sec
Iteration 35: 0.011 sec
Iteration 36: 0.011 sec
Iteration 37: 0.011 sec
Iteration 38: 0.011 sec
Iteration 39: 0.011 sec
Iteration 40: 0.011 sec
Average time: 0.011 sec
Batch size = 1
Latency: 10.987 ms
Throughput: 91.020 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint=None, data_location=None, framework='tensorflow', input_graph='/in_graph/freezed_resnet50.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir=None, num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/eval_image_classifier_inference.py --input-graph=/in_graph/freezed_resnet50.pb --model-name=resnet50 --inter-op-parallelism-threads=1 --intra-op-parallelism-threads=28 --batch-size=1
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --use-case=image_recognition         --model-name=resnet50         --platform=fp32         --mode=inference         --intelai-models=/workspace/intelai_models         --batch-size=1         --single-socket         --verbose         --in-graph=/in_graph/freezed_resnet50.pb
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log
```

* To measure the model Throughput, set `--batch-size=128` and run the benchmark script as shown:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/myuser/resnet50_fp32_pretrained_model/freezed_resnet50.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --batch-size=128 \
    --single-socket \
    --docker-image docker_image \
    --verbose
```

The log file is saved to:
`models/benchmarks/common/tensorflow/logs`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 1.765 sec
Iteration 2: 0.640 sec
Iteration 3: 0.638 sec
Iteration 4: 0.637 sec
Iteration 5: 0.632 sec
Iteration 6: 0.631 sec
Iteration 7: 0.632 sec
Iteration 8: 0.635 sec
Iteration 9: 0.634 sec
Iteration 10: 0.641 sec
Iteration 11: 0.632 sec
Iteration 12: 0.640 sec
Iteration 13: 0.632 sec
Iteration 14: 0.632 sec
Iteration 15: 0.635 sec
Iteration 16: 0.634 sec
Iteration 17: 0.633 sec
Iteration 18: 0.631 sec
Iteration 19: 0.631 sec
Iteration 20: 0.633 sec
Iteration 21: 0.637 sec
Iteration 22: 0.630 sec
Iteration 23: 0.631 sec
Iteration 24: 0.635 sec
Iteration 25: 0.635 sec
Iteration 26: 0.641 sec
Iteration 27: 0.632 sec
Iteration 28: 0.629 sec
Iteration 29: 0.631 sec
Iteration 30: 0.631 sec
Iteration 31: 0.635 sec
Iteration 32: 0.632 sec
Iteration 33: 0.635 sec
Iteration 34: 0.631 sec
Iteration 35: 0.635 sec
Iteration 36: 0.631 sec
Iteration 37: 0.635 sec
Iteration 38: 0.632 sec
Iteration 39: 0.631 sec
Iteration 40: 0.630 sec
Average time: 0.633 sec
Batch size = 128
Throughput: 202.190 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=False, checkpoint=None, data_location=None, framework='tensorflow', input_graph='/in_graph/freezed_resnet50.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir=None, num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/eval_image_classifier_inference.py --input-graph=/in_graph/freezed_resnet50.pb --model-name=resnet50 --inter-op-parallelism-threads=1 --intra-op-parallelism-threads=28 --batch-size=128
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --use-case=image_recognition         --model-name=resnet50         --platform=fp32         --mode=inference         --intelai-models=/workspace/intelai_models         --batch-size=128         --single-socket         --verbose         --in-graph=/in_graph/freezed_resnet50.pb
Batch Size: 128
Ran inference with batch size 128
Log location outside container: home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log
```