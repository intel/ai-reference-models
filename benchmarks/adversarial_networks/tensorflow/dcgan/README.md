# DCGAN

This document has instructions for how to run DCGAN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference.

## FP32 Inference Instructions

1. Clone the `tensorflow/models` repository:

```
$ git clone https://github.com/tensorflow/models.git
```

The TensorFlow models repo will be used for running inference as well as
converting the CIFAR-10 dataset to the TF records format.

2. Follow the TensorFlow models 
[Generative Adversarial Network](https://github.com/tensorflow/models/tree/master/research/gan#cifar10) (GAN)
[instructions](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_cifar10.py)
to download and convert the CIFAR-10 dataset.

3. A link to download the pre-trained model is coming soon.

4. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the DCGAN model code.

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 4.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the external model directory
for `--model-source-dir` (from step 1) `--data-location` (from step 2), and `--checkpoint` (from step 3).


Run benchmarking for throughput and latency with `--batch-size=100` :
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name dcgan \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/myuser/cifar10 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

5. Log files are located at the value of `--output-dir`.

Below is a sample log file tail when running benchmarking for throughput:
```
Batch size: 100 
Batches number: 500
Time spent per BATCH: 35.8268 ms
Total samples/sec: 2791.2030 samples/s
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_dcgan_inference_fp32_20190117_220342.log
```