# Object Detection with TensorFlow Serving on CPU
### Real-time and Max Throughput Inference
Model: R-FCN

## Goal

This tutorial will introduce you to the CPU performance considerations for object detection deep learning models and
how to use Intel’s optimizations for [TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPU. 
It also provides sample code that you can use to get your optimized TensorFlow model server and REST client up and running quickly.

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](https://github.com/NervanaSystems/intel-models/blob/master/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](https://github.com/NervanaSystems/intel-models/blob/master/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * **Performance Metrics**
   * **TensorFlow Serving Configuration Settings**
* Ran an example end-to-end using a REST client, such as the one in the [Installation Guide](https://github.com/NervanaSystems/intel-models/blob/master/docs/general/tensorflow_serving/InstallationGuide.md)
  
## Background

The Intel MKL-DNN (Math Kernel Library for Deep Neural Networks) offers significant performance improvements for convolution, pooling, normalization, activation, and other operations for object detection, using efficient vectorization and multi-threading.
Tuning TensorFlow Serving to take full advantage of your hardware for object detection deep learning inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a REST client notebook to verify object detection and measure latency and throughput
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial - R-FCN

For steps 1 and 2, refer to the Intel Model Zoo FP32 benchmarks:
* [R-FCN README](https://github.com/NervanaSystems/intel-models/tree/master/benchmarks/object_detection/tensorflow/rfcn#fp32-inference-instructions)

1. **Download the Model**: Download and extract the R-FCN pre-trained model (FP32), using the instructions in the README above.

2. **Download Data**: Follow the instructions in the README above to download the COCO dataset.
   **Note**: Do not convert the COCO dataset to TF records format.

3. **Clone this repository**: Clone the intel-models repo and `cd` into the `docs/object_detection/tensorflow_serving` directory.

4. **Set up your environment**: In this tutorial, we use a virtual environment to install the required packages. 
   If you do not have pip or virtualenv, you will need to get them first:
   ```
   $ sudo apt-get install python-pip
   $ pip install virtualenv
   $ virtualenv venv
   ```
   Then activate the virtual environment and install `requests`, `jupyter`, `matplotlib`, and `pillow`:
   ```
   $ source venv/bin/activate
   (venv)$ pip install requests
   (venv)$ pip install jupyter
   (venv)$ pip install matplotlib
   (venv)$ pip install pillow
   ```
   In addition, we need the object detection API from the Tensorflow models repo. 
   Follow [these steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install it (this will prompt you to install tensorflow and a few more pip packages also).
   
5. **Prepare the SavedModel**: After you extract the model zip file, there will be a `saved_model` folder with a `saved_model.pb` file in it.
   Create a directory `/tmp/rfcn/1` and copy the `saved_model.pb` file to that location (the `1` subdirectory is important - don't skip it!). 
   This is the file we will serve from TensorFlow Serving. For more information about SavedModels, see these references:
   * [SavedModel](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model)
   * [SignatureDefs](https://www.tensorflow.org/serving/signature_defs) 
   
6. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`.
   For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, *num_physical_cores* = 28 * 2 = 56.

7. **Start the server**: Now let's start up the TensorFlow model server. To optimize overall performance, use the following recommended settings from the
   [General Best Practices](https://github.com/NervanaSystems/intel-models/blob/master/docs/general/tensorflow_serving/GeneralBestPractices.md):
   * OMP_NUM_THREADS = *num_physical_cores*
   * TENSORFLOW_SESSION_PARALLELISM = *num_physical_cores*/4
   
   For our example with 56 physical cores, these values are 56 and 14:
   ```
   (venv)$ docker run --name=tfserving --rm -d -p 8501:8501 -v "/tmp/rfcn:/models/rfcn" -e MODEL_NAME=rfcn -e OMP_NUM_THREADS=56 -e TENSORFLOW_SESSION_PARALLELISM=14 tensorflow/serving:mkl
   ```
   Note: For some models, playing around with these settings values can improve performance even further. 
   We are exploring approaches to fine-tuning the parameters and will present our findings in a future version of this document. 
   We recommend that you experiment with your own hardware and model if you have strict performance requirements.

8. **Run a Test**: Now we can run a Jupyter notebook that selects a COCO file and sends it for object detection.
   Run `jupyter notebook`, open the RFCN.ipynb file, and execute the cells in order. 
   The output of the "Test Object Detection" section should be an image with objects correctly detected by the R-FCN model.

9. **Real-time inference**: Real-time inference is measured by latency and is usually defined as batch size 1.
   To see average inference latency (in ms), continue running the Jupyter notebook through the section titled "Real-time Inference" using batch_size 1.
   
   In some cases, it is desirable to constrain the inference server to a single core or socket. 
   Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources.
   See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for all the options and their definitions.
   For example, to run the container so that a single CPU is used, you can use these settings:
   * --cpuset-cpus = "0"
   * --cpus = "1"
   * OMP_NUM_THREADS = 1
   * TENSORFLOW_SESSION_PARALLELISM = 1
   ```
   (venv)$ docker run --name=tfserving --rm --cpuset-cpus="0" --cpus="1" -d -p 8500:8500 -v "/tmp/rfcn:/models/rfcn" -e MODEL_NAME=rfcn -e OMP_NUM_THREADS=1 -e TENSORFLOW_SESSION_PARALLELISM=1 tensorflow/serving:mkl
   ```

10. **Maximum throughput**: Regardless of hardware, the best batch size for throughput is 128. 
    To see average throughput (in images/sec), continue running the Jupyter notebook through the section titled "Throughput" using batch_size 128.
    
11. **Clean up**: 
    * After saving any changes you made to the Jupyter notebook, close the file and stop the Jupyter server by clicking `Quit` from the main file browser. 
    * To shut off the server, stop the docker process that is running it. View your running containers with `docker ps`.
      To stop one, copy the Container ID and run `docker stop <container_id>`.
    * Deactivate your virtual environment with `deactivate`.
    
## Conclusion

You have now seen an end-to-end example of serving an object detection model for inference using TensorFlow Serving, and learned:
1. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
2. How to verify that the served model can correctly detect objects in an image using a sample Jupyter notebook
3. How to benchmark latency and throughput metrics using a REST client

With this knowledge and the example code provided, you should be able to get started serving your own custom object detection model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement are possible.
