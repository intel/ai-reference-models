# ResNet50 (v1.5)

This document has instructions for how to run ResNet50 (v1.5) for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

2. Download the pre-trained model.
```
$ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```

3. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a tensorflow serving run using optimized TensorFlow Serving docker
container. It has arguments to specify which model, framework, mode,
precision, input graph and docker image.

Substitute in your own `--in-graph` pretrained model file path (from step 2).

4. For specifying a docker image that the model server should run with, you should use the `--docker-image` arg. This will pull the TensorFlow Serving image and run the model in docker. The client benchmarking script will then be launched from a virtualenv on bare metal and make requests to the serving container over gRPC.

5. ResNet50 (v1.5) can be run for measuring batch or online inference performance. Use one of the following examples below,
depending on your use case.

* For online inference with dummy data (using `--batch-size 1`):

```
python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow_serving \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
    --benchmark-only
```
Example log tail when running for online inference:
```
Iteration 35: ... sec
Iteration 36: ... sec
Iteration 37: ... sec
Iteration 38: ... sec
Iteration 39: ... sec
Iteration 40: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
tfserving_9350
Log output location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_20200805_103016.log
```

* For batch inference with dummy data (using `--batch-size 128`):

```
python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow_serving \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
    --benchmark-only
```
Example log tail when running for batch inference:
```
Iteration 34: ... sec
Iteration 35: ... sec
Iteration 36: ... sec
Iteration 37: ... sec
Iteration 38: ... sec
Iteration 39: ... sec
Iteration 40: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
tfserving_23640
Log output location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_20200805_104341.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.
