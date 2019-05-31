# General Best Practices for Intel® Optimization of TensorFlow Serving

## Introduction

The Intel® Optimization of TensorFlow Serving leverages Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) to perform inferencing tasks significantly faster than the default installation on Intel® processors.
To install Intel Optimization of TensorFlow Serving, please refer to the [TensorFlow Serving Installation Guide](InstallationGuide.md).
Here we provide a brief description of the available TensorFlow model server settings and some general best practices for optimal performance.
Due to potential differences in environment, memory, network, model topology, and other factors,
we recommend that you explore a variety of settings and hand-tune TensorFlow Serving for your particular workload and metric of interest,
but the following information will help you get started.

## Performance Metrics

* **Batch Inference** measures how many input tensors can be processed per second with batches of size greater than one.
Typically for batch inference, optimal performance is achieved by exercising all the physical cores on a socket.
* **Online Inference** (also called real-time inference) is a measurement of the time it takes to process a single input tensor, i.e. a batch of size one.
In a real-time inference scenario, optimal performance is achieved by minimizing thread launching and orchestration between concurrent processes.
This guide will help you set your TensorFlow model server options for good balanced performance over both metrics.
However, if you want to prioritize one metric over the other or further tune TensorFlow Serving for your specific model, see the [tutorials](/docs#tutorials-by-use-case).

## TensorFlow Serving Configuration Settings

There are four parameters you can set when running the TensorFlow Serving with Intel MKL-DNN docker container.
* ***OMP_NUM_THREADS*** is the maximum number of threads available. A good guideline is to set it equal to the number of physical cores.
* ***TENSORFLOW_INTER_OP_PARALLELISM*** is the number of thread pools to use for a TensorFlow session. A good guideline we have found empirically is to set this to 2 (you may want to start with this suggestion but then try other values, as well).
* ***TENSORFLOW_INTRA_OP_PARALLELISM*** is the number of threads in each thread pool to use for a TensorFlow session. A good guideline is to set it equal to the number of physical cores. 
The number of physical cores (referred to from now on as *num_physical_cores*) may be different from the number of logical cores or CPUs and can be found in Linux with the `lscpu` command.
* ***(DEPRECATED) TENSORFLOW_SESSION_PARALLELISM*** is the number of threads to use for a TensorFlow 1.12 session. This controls both intra-op and inter-op parallelism and has been replaced by the separate parameters in TensorFlow Serving 1.13.
There is backward compatibility for ***TENSORFLOW_SESSION_PARALLELISM*** for all versions, but if you use it, both inter- and intra-op parallelism will be set to the same value, which is usually not optimal. See this [feature description](https://github.com/tensorflow/serving/pull/1253) for the full logic. 
If you are using version 1.12, a good guideline is to set this parameter equal to one-quarter the number of physical cores, but for 1.13 and above, we recommend omitting it and using ***TENSORFLOW_INTRA_OP_PARALLELISM*** and ***TENSORFLOW_INTER_OP_PARALLELISM*** instead.

### Example

To compute *num_physical_cores*, execute the `lscpu` command and multiply `Core(s) per socket` by `Socket(s)`.
For example, *num_physical_cores* = 16 * 1 = 16 for a machine with this `lscpu` output:
```
$lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                32
On-line CPU(s) list:   0-31
Thread(s) per core:    2
Core(s) per socket:    16
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) CPU @ 2.00GHz
Stepping:              3
CPU MHz:               2000.168
BogoMIPS:              4000.33
Hypervisor vendor:     KVM
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              56320K
NUMA node0 CPU(s):     0-31
Flags:                 fpu vme de pse tsc ...
```
Next, compute ***OMP_NUM_THREADS***, ***TENSORFLOW_INTER_OP_PARALLELISM***, and ***TENSORFLOW_INTRA_OP_PARALLELISM***:
  * ***OMP_NUM_THREADS*** = *num_physical_cores* = 16
  * ***TENSORFLOW_INTER_OP_PARALLELISM*** = 2
  * ***TENSORFLOW_INTRA_OP_PARALLELISM*** = num_physical_cores = 16
  
Then start the model server container from the TensorFlow Serving docker image with a command that sets the environment variables to these values
(this assumes some familiarity with docker and the `docker run` command):
```
docker run \
    --name=tfserving_mkl \
    --rm \
    -d \
    -p 8500:8500 \
    -v "/home/<user>/<saved_model_directory>:/models/<model_name>" \
    -e MODEL_NAME=<model_name> \
    -e OMP_NUM_THREADS=16 \
    -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
    -e TENSORFLOW_INTRA_OP_PARALLELISM=16 \
    tensorflow/serving:mkl
```

## Data Format

Data format can play an important role in inference performance. For example, the multi-dimensional image data inputs for image recognition and object detection models can be stored differently in memory address space.
Image data format can be represented by 4 letters for a 2-dimensional image - N: number of images in a batch, C: number of channels in an image, W: width of an image in pixels, and H: height of an image in pixels.
The order of the four letters indicates how pixel data are stored in 1-dimensional memory space, with the outer-most dimension first and the inner-most dimension last.
Therefore, NCHW indicates pixel data are stored first width-wise (the inner-most dimension), followed by height, then channel, and finally batch (Figure 1).
TensorFlow supports both NCHW and NHWC data formats. While NHWC is the default format, NCHW is more efficient for Intel Optimization of TensorFlow Serving with Intel MKL-DNN and is the recommended layout for image data.

![NCHW format](nchw.png)

Figure 1. NCHW format

## Summary of General Best Practices

1. Find *num_physical_cores* by using the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`
2. Run the TensorFlow model server docker container with:
    - ***OMP_NUM_THREADS*** = *num_physical_cores*
    - ***TENSORFLOW_INTER_OP_PARALLELISM*** = 2
    - ***TENSORFLOW_INTRA_OP_PARALLELISM*** = *num_physical_cores*
3. Use NCHW data format for images
4. See one of the hands-on [tutorials](/docs/README.md) for an advanced walkthrough of your use case
