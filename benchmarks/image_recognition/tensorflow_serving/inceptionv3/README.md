# Inception V3

This document has instructions for how to run Inception V3 for the
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
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb
```

3. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a tensorflow serving run using optimized TensorFlow Serving docker
container. It has arguments to specify which model, framework, mode,
precision, input graph and docker image.

Substitute in your own `--in-graph` pretrained model file path (from step 2).

4. For specifying a docker image that the model server should run with, you should use the `--docker-image` arg. This will pull the TensorFlow Serving image and run the model in docker. The client benchmarking script will then be launched from a virtualenv on bare metal and make requests to the serving container over GRPC.

5. Inception V3 can be run for measuring batch or online inference performance. Use one of the following examples below,
depending on your use case.

* For online inference with dummy data (using `--batch-size 1`):

```
python launch_benchmark.py \
    --in-graph /home/<user>/inceptionv3_fp32_pretrained_model.pb \
    --model-name inceptionv3 \
    --framework tensorflow_serving \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
    --benchmark-only
```
Example log tail when running for online inference:
```
Iteration 35: 0.019 sec
Iteration 36: 0.020 sec
Iteration 37: 0.018 sec
Iteration 38: 0.018 sec
Iteration 39: 0.019 sec
Iteration 40: 0.018 sec
Average time: 0.019 sec
Batch size = 1
Latency: 18.801 ms
Throughput: 53.189 images/sec
tfserving_3784
Log output location: {--output-dir value}/benchmark_inceptionv3_inference_fp32_20190516_103531.log
```

* For batch inference with dummy data (using `--batch-size 128`):

```
python launch_benchmark.py \
    --in-graph /home/<user>/inceptionv3_fp32_pretrained_model.pb \
    --model-name inceptionv3 \
    --framework tensorflow_serving \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
    --benchmark-only
```
Example log tail when running for batch inference:
```
Iteration 34: 0.779 sec
Iteration 35: 0.916 sec
Iteration 36: 0.809 sec
Iteration 37: 0.793 sec
Iteration 38: 0.813 sec
Iteration 39: 0.796 sec
Iteration 40: 0.796 sec
Average time: 0.817 sec
Batch size = 128
Throughput: 156.752 images/sec
tfserving_5299
Log output location: {--output-dir value}/benchmark_inceptionv3_inference_fp32_20190516_103958.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.
