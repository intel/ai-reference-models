# MLPerf GNMT

This document has instructions for how to run GNMT for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instruction

1. Clone the intelai/models repo.
This repo has the launch script for running the model, which we will
use in the next step.
```
git clone https://github.com/IntelAI/models.git
```

2. Download GNMT benchmarking data.
```
wget https://zenodo.org/record/2531868/files/gnmt_inference_data.zip
unzip gnmt_inference_data.zip
```

3. Download the pretrained model:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mlperf_gnmt_fp32_pretrained_model.pb
```

4. Install Intel TensorFlow:
 It's a main dependency to build TensorFlow addons repository and create a pip wheel.
```
pip install intel-tensorflow==2.3.0
```

Clone TensorFlow addons (r0.5) and apply a patch:

A patch file is attached in Intel Model Zoo MLpref GNMT model scripts, it fixes TensorFlow addons (r0.5) to work with TensorFlow version 2.3,
and prevents TensorFlow 2.0.0 to be installed by default as a required dependency.
```
git clone --single-branch --branch=r0.5 https://github.com/tensorflow/addons.git
cd addons
git apply /home/<user>/models/models/language_translation/tensorflow/mlperf_gnmt/gnmt-v0.5.2.patch
```

Build TensorFlow addons source code and create TensorFlow addons pip wheel:
```
bash configure.sh

bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
cp artifacts/tensorflow_addons-*.whl /home/<user>/models/models/language_translation/tensorflow/mlperf_gnmt
```
>Note: for running on bare metal, please install the `tensorflow_addons` wheel on your machine:
```
pip install artifacts/tensorflow_addons-*.whl --no-deps
```

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.

Substitute in your own `--data-location` (from step 2), `--in-graph` pre-trained
model file path (from step 3).

For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
```
python launch_benchmark.py \
--model-name mlperf_gnmt \
--framework tensorflow \
--precision fp32 \
--mode inference \
--batch-size 1 \
--socket-id 0 \
--data-location /home/<user>/nmt/data \
--docker-image intel/intel-optimized-tensorflow:2.3.0 \
--in-graph /home/<user>/mlperf_gnmt_fp32_pretrained_model.pb \
--benchmark-only
```

For batch inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 32`):
```
python launch_benchmark.py \
--model-name mlperf_gnmt \
--framework tensorflow \
--precision fp32 \
--mode inference \
--batch-size 32 \
--socket-id 0 \
--data-location /home/<user>/nmt/data \
--docker-image intel/intel-optimized-tensorflow:2.3.0 \
--in-graph /home/<user>/mlperf_gnmt_fp32_pretrained_model.pb \
--benchmark-only
```

For accuracy test (using `--accuracy_only`, `--socket-id 0` and `--batch-size 32`):
```
python launch_benchmark.py \
--model-name mlperf_gnmt \
--framework tensorflow \
--precision fp32 \
--mode inference \
--batch-size 32 \
--socket-id 0 \
--data-location /home/<user>/nmt/data \
--docker-image intel/intel-optimized-tensorflow:2.3.0 \
--in-graph /home/<user>/mlperf_gnmt_fp32_pretrained_model.pb \
--accuracy-only
```
