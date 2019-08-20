# DCGAN

This document has instructions for how to run DCGAN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Script instructions for model training and inference.

## FP32 Inference Instructions

1. Store the path to the current directory and clone the `tensorflow/models` repository into tf_models:

```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR

$ git clone https://github.com/tensorflow/models.git tf_models
$ cd tf_models/
$ git checkout -b r1.11 remotes/origin/r1.11
```

The TensorFlow models repo will be used for running inference as well as
converting the CIFAR-10 dataset to the TF records format.

2. Follow the TensorFlow models 
[Generative Adversarial Network](https://github.com/tensorflow/models/tree/master/research/gan#cifar10) (GAN)
[instructions](https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_cifar10.py)
to download and convert the CIFAR-10 dataset.

3. Download and extract the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/dcgan_fp32_unconditional_cifar10_pretrained_model.tar.gz
$ tar -xvf dcgan_fp32_unconditional_cifar10_pretrained_model.tar.gz
```

4. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running an optimized version of the DCGAN model code.

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 4.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model script run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the external model directory
for `--model-source-dir` (from step 1) `--data-location` (from step 2), and `--checkpoint` (from step 3).


Run the model script for batch and online inference with `--batch-size=100` :
```
$ cd models/benchmarks

$ python launch_benchmark.py \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name dcgan \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 100 \
    --socket-id 0 \
    --checkpoint $MODEL_WORK_DIR/dcgan_fp32_unconditional_cifar10_pretrained_model \
    --data-location $MODEL_WORK_DIR/cifar10 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14
```

5. Log files are located at the value of `--output-dir`.

Below is a sample log file tail when running for batch inference:
```
Batch size: 100 
Batches number: 500
Time spent per BATCH: 35.8268 ms
Total samples/sec: 2791.2030 samples/s
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_dcgan_inference_fp32_20190117_220342.log
```

6. To return to where you started from:
```
$ popd
```
