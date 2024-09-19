# Quantizing a Pre-trained Model from Model Zoo for Intel® Architecture with Intel® Neural Compressor


## Background

Low-precision inference can speed up inference, by converting the fp32 model to int8 or bf16 model. Intel provides Intel® Deep Learning Boost technology in the Second Generation Intel® Xeon® Scalable Processors and newer Xeon®, which supports to speed up int8 and bf16 model by hardware.

Intel® Neural Compressor helps the user to simplify the processing to convert the fp32 model to int8/bf16.

At the same time, Intel® Neural Compressor will tune the quanization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is released in Intel® AI Analytics Toolkit and works with Intel® Optimization of TensorFlow*.

Please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Purpose

Demostrate how to quantize a pre-trained model from Model Zoo for Intel(R) Architecture with Intel® Neural Compressor and show the performance gain after quantization.

We will learn the acceleration of AI inference by Intel AI technology:

1. Intel® Deep Learning Boost

2. Intel® Neural Compressor

3. Intel® Optimization for Tensorflow*

## Prerequisites

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 or newer
| Hardware                          | The Second Generation Intel(R) Xeon(R) Scalable processor family or newer (support AVX2, AVX512 or Intel(R) Deep Learning Boost).
| Software                          | Intel® Neural Compressor, Intel Optimization for Tensorflow 2.5 (or newer)
| What you will learn               | How to use Intel® Neural Compressor to quantize a pre-trained TensorFlow model and achieve better inference performance on Intel(R) Xeon(R) CPU
| Time to complete                  | 1-2+ hours

## Running Environment

### Running on Devcloud

Follow the [Prepare Software Environment](#prepare-software-environment) section to set up the Python environment on DevCloud. For more information, refer to the [Intel(R) oneAPI AI Analytics Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get-started/analytics-toolkit/)

### Running in a Local Machine

Follow the "Prepare Software Environment" section to setup the Python environment and ensure that the prerequisites are in comply with your local machine.

## Prepare Software Environment


### Install Python Running Time (optional):


```
sudo apt-get update && apt-get install -y  build-essential python3-opencv python3-dev wget vim
```

### Setup Python Virtual Environment

```
./set_env.sh
```

## Files and folders structure


| File/Folder | Description
|:---                               |:---
|format2json.py| A Translation Script to convert the result of benchmark from text format to JSON format.
|local_benchmark.sh| A bash script to test the model benchmark (Throughput, Latency & Accuracy). It is wrapper layer for launch_benchmark.py provided by Model Zoo for Intel(R) Architecture.
|inc_for_tensorflow.ipynb| A Jupyter notebook file for this tutorial.
|inc_quantize_model.py| A python script to quantize the FP32 model to INT8 by calling Intel® Neural Compressor API.
|pyproject.toml| List all of packages needed by this sample.
|resnet50_v1.yaml| A default YAML file for Intel® Neural Compressor. No update needed.
|run_jupyter.sh| A script to start running Jupyter notebook.
|tf_2012_val | A default folder to save the dataset. Put the TFRecord format dataset files into it.
|tips.md| An addtional document for some tips on Intel® Neural Compressor usage.
|ut.sh| A Unit test script. The dataset should be saved in tf_2012_val folder befure users run the script.


## How to Run

### Start running Jupyter Notebook


Steps:

```
./run_jupyter.sh

(tensorflow) xxx@yyy:$ [I 09:48:12.622 NotebookApp] Serving notebooks from local directory:
...
[I 09:48:12.622 NotebookApp] Jupyter Notebook 6.1.4 is running at:
[I 09:48:12.622 NotebookApp] http://yyy:8888/?token=<token>
[I 09:48:12.622 NotebookApp]  or http://127.0.0.1:8888/?token=<token>
[I 09:48:12.622 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:48:12.625 NotebookApp]

    To access the notebook, open this file in a browser:
        ...
    Or copy and paste one of these URLs:
        http://yyy:8888/?token=<token>
     or http://127.0.0.1:8888/?token=<token>
[I 09:48:26.128 NotebookApp] Kernel started: bc5b0e60-058b-4a4f-8bad-3f587fc080fd, name: python3
```

### Open Jupyter Notebook

Open link: **http://yyy:8888** in a web browser and enter the token generated during Jupyter startup. Click 'inc_for_tensorflow.ipynb'.

### Choose Kernel

Choose the env_inc kernel in the menu: Kernel -> Change kernel -> env_inc

### Run

Run all the cells in Jupyter Notebook one by one.

## Unit Test

Please prepare the ImageNet validation dataset and save it into **tf_2012_val** folder.

Execute `./ut.sh` to run unit tests.

It returns 0 if there is no error.


## Additional Information
Please refer to the [Intel® Neural Compressor document](https://intel.github.io/neural-compressor/README.html) for further details.

Some [tips](tips.md) to reduce the data processing time are also provided for this tutorial .
