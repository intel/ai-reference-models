# General Best Practices for Intel® Optimized TensorFlow

## Introduction

To get the highest possible performance on Intel Architecture (IA) and thus yield high performance, TensorFlow can be powered by Intel® with highly optimized math routines for deep learning tasks. This primitives library is called Intel Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN). Intel® MKL-DNN includes convolution, normalization, activation, inner product and other primitives. 
Please see the [install guide](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide) for how to setup Intel® Optimization of Tensorflow on your system. 
The Bottom Line: The user gets accelerated CPU TF execution with no code changes.  



## Performance Metrics

* **Throughput** measures how many input tensors can be processed per second with batches of size greater than one.
Typically for maximum throughput, optimal performance is achieved by exercising all the physical cores on a socket.
* **Latency** (also called real-time inference) is a measurement of the time it takes to process a single input tensor, i.e. a batch of size one.
In a real-time inference scenario, optimal latency is achieved by minimizing thread launching and orchestration between concurrent processes.

This guide will help you set your TensorFlow runtime options for good balanced performance over both metrics.
However, if you want to prioritize one metric over the other or further tune Tensorflow for your specific model, please see the tutorials. A link to these can be found in the [Model Zoo docs readme](/docs/README.md).

## TensorFlow Configuration Settings

These are the parameters you need to set when running TensorFlow with MKL-DNN. A more complete description of these settings can be found in the [performance considerations article](https://software.intel.com/en-us/articles/maximize-TensorFlow-performance-on-cpu-considerations-and-recommendations-for-inference).  

### TensorFlow Runtime Settings

* ***inter_op_parallelism_threads*** is the number of thread pools to use for a TensorFlow session. A good guideline we have found empirically is to set this to *2*. (you may want to start with this suggestion but then try other values, as well).

* ***intra_op_parallelism_threads*** is the number of threads in each threadpool to use for a TensorFlow session. This should be set to the number of physical cores  may be different from the number of logical cores or CPUs and can be found in Linux with the `lscpu` command.

* ***Data Format*** specifies the way data is stored and accesed in memory. We recommend using channels-first (NCHW) format. Please see the [data format section of performance doc](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference#inpage-nav-2-2) for more information.  

### Environment Variables

* ***OMP_NUM_THREADS*** is the maximum number of threads available for the OpenMP runtime. A good guideline is to set it equal to the number of physical cores.
* ***KMP_BLOCKTIME*** This environment variable sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. The default value is 200ms. A small KMP_BLOCKTIME value may offer better overall performance if application contains non-OpenMP threaded code. A larger KMP_BLOCKTIME value may be more appropriate if threads are to be reserved solely for use for OpenMP execution. It is suggested to be set to 0 for CNN based models. KMP_BLOCKTIME=1 is a goood place to start for non-CNN topologies. 
* ***KMP_AFFINITY*** controls how threads are distributed and ultimately bound to specific processing units. *KMP_AFFINITY=granularity=fine,verbose,compact,1,0* is recommended.

### NUMA Runtime

* ***cpu_node_bind/membind*** It is recommended to confine a TensorFlow session to a single NUMA node. If numa is enabled on your system, use *numactl --cpunodebind=0 --membind=0 python* to call your TensorFlow script. 
* ***Concurrent Execution*** You can fully exercise your hardware by sharding your data and launching multiple executions concurrenly, each bound to a different NUMA node. To do this, use the *&* command to launch non-blocking execution in the shell: *numactl --cpunodebind=0 --membind=0 python & numactl --cpunodebind=1 --membind=1 python*


### lscpu

To help set these settings, you can execute the `lscpu` command in linux to find important information about your system:
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

To get number of physical cores, simple multiply `Core(s) per socket` by `Socket(s)`.

For usage specifics and command line examples, see one of the hands-on tutorials for an advanced walkthrough of your use case:
   * Coming Soon, See [README](/docs/README.md) for details.