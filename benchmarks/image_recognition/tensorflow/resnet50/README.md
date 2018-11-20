# ResNet50

This document has instructions for how to run ResNet50 for the
following platforms:
* [Int8 inference](#int8-inference-instructions)

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
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb
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
the pre-trained `resnet50_int8_pretrained_model.pb` input graph file (from step
2, the docker image (from step 3) and the `--accuracy-only` flag.
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/dataset/FullImageNetData_directory
    --in-graph /home/myuser/resnet50_int8_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --platform int8 \
    --mode inference \
    --batch-size=100 \
    --accuracy-only \
    --docker-image docker_image
```
The log file is saved to:
`models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log`.

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
Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resnet50_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100 --data-location=/dataset --single-socket --verbose --accuracy-only                          --in-graph=/in_graph/resnet50_int8_pretrained_model.pb
Batch Size: 100
Ran inference with batch size 100
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log
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
Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resnet50_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet50', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet50 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --data-location=/dataset --single-socket --verbose              --benchmark-only             --in-graph=/in_graph/resnet50_int8_pretrained_model.pb 
Batch Size: 128
Ran inference with batch size 128
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference.log

```