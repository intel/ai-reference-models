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
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mlperf_gnmt_fp32_pretrained_model.pb
```

4. Please ensure you have installed the libraries listed in the
`requirements.txt` before you start the next step.
Clone tensorflow-addons repo
```
pip install intel-tensorflow==2.1.0
git clone -b v0.5.2 https://github.com/tensorflow/addons.git
cd addons
sed -i 's;\${PYTHON_VERSION:=python} -m pip install $QUIET_FLAG -r $REQUIREMENTS_TXT;PYTHON_VERSION=python;' configure.sh
sh configure.sh
bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install artifacts/tensorflow_addons-*.whl
```

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.

Substitute in your own `--data-location` (from step 2), `--checkpoint` pre-trained
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
--in-graph /home/<user>/mlperf_gnmt_fp32_pretrained_model.pb \
--accuracy_only
```
