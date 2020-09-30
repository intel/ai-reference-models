# Image Recognition with TensorFlow Serving on CPU

### Online and Batch Inference
Model and Precision: InceptionV3 FP32, ResNet50 FP32, and ResNet50 Int8

## Goal

This tutorial will introduce you to the CPU performance considerations for image recognition deep learning models with different precisions and
how to use Intel® Optimizations for [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) to improve inference time on CPUs. 
It also provides sample code that you can use to get your optimized TensorFlow model server and gRPC client up and running quickly.

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * **Performance Metrics**
   * **TensorFlow Serving Configuration Settings**
* Ran an example end-to-end using a gRPC client, such as the one in the [Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md#option-2-query-using-grpc)

> Note: We use gRPC in this tutorial and offer another [tutorial](/docs/object_detection/tensorflow_serving/Tutorial.md) that illustrates the use of the REST API if you are interested in that protocol. 

## Background

Convolutional neural networks (CNNs) for image recognition are computationally expensive. 
The Intel® oneAPI Deep Neural Network Library (Intel® oneDNN) offers significant performance improvements for convolution, pooling, normalization, activation, and other operations via efficient vectorization and multi-threading.
Tuning TensorFlow Serving to take full advantage of your hardware for image recognition deep learning inference involves:
1. Working through this tutorial to set up servable versions of the well-known [ResNet50](https://arxiv.org/pdf/1512.03385.pdf) and [InceptionV3](https://arxiv.org/pdf/1512.00567v1.pdf) CNN models with different precisions.
2. Running a TensorFlow Serving docker container configured for performance given your hardware resources
3. Running a client script to measure online and batch inference performance
4. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial - InceptionV3 and Resnet50

This section shows a step-by-step example for how to serve one of the following Image Recognition models
`(ResNet50 FP32, ResNet50 Int8, and InceptionV3 FP32)` using TensorFlow Serving.
It also explains how to manage the available CPU resources and tune the model server for the optimal performance.

For steps 1 and 2, refer to the model READMEs for the most up-to-date pre-trained model URLs and dataset instructions:
    * [InceptionV3 FP32 README](/benchmarks/image_recognition/tensorflow/inceptionv3#fp32-inference-instructions) 
    * [ResNet50 FP32 README](/benchmarks/image_recognition/tensorflow/resnet50#fp32-inference-instructions)
    * [ResNet50 Int8 README](/benchmarks/image_recognition/tensorflow/resnet50#int8-inference-instructions)

>NOTE: The below example shows InceptionV3 (FP32). The same code snippets will work for ResNet50 (FP32 and Int8) by replacing the model name to `resnet50`.

1. **Download the Model**: Download and extract the pre-trained model.
   ```
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb
   ```

2. **(Optional) Download Data**: If you are interested only in testing performance, not accuracy, you can skip this step and use synthetic data.
   If you want to verify prediction accuracy by testing on real data, follow the instructions in one of the READMEs above to download the ImageNet dataset.

3. **Clone this repository**: Clone the [intelai/models](https://github.com/intelai/models) repository and `cd` into the `models/benchmarks/image_recognition/tensorflow_serving/inceptionv3/inference/fp32` directory.
   ```
   git clone https://github.com/IntelAI/models.git
   cd models/benchmarks/image_recognition/tensorflow_serving/inceptionv3/inference/fp32
   ```

4. **Set up your environment**: In this tutorial, we use a virtual environment to install a few required Python packages. 
   If you do not have pip or virtualenv, you will need to get them first:
   ```
   sudo apt-get install python-pip
   pip install virtualenv
   virtualenv -p python3 venv
   ```
   Then activate the virtual environment and install `requests` and `tensorflow-serving-api`:
   ```
   source venv/bin/activate
   pip install requests tensorflow-serving-api
   ```
5. **Create a SavedModel**: Using the conversion script `model_graph_to_saved_model.py`, convert the pre-trained model graph to a SavedModel.
   (For ResNet50, substitute the name of the ResNet50 FP32 or the ResNet50 Int8 pre-trained model.)
   
   Example:
   ```
   python model_graph_to_saved_model.py --import_path inceptionv3_fp32_pretrained_model.pb
   ```
   
   Console out:
   ```
   ...
   I0720 14:54:02.212594 139773959747392 builder_impl.py:426] SavedModel written to: /tmp/1/saved_model.pb
   Done!
   ```
   This will create a `/tmp/1/` directory with a `saved_model.pb` file in it. This is the file we will serve from TensorFlow Serving.
   The `model_graph_to_saved_model.py` script has applied some transform optimizations and attached a signature definition to the model
   in order to make it compatible with TensorFlow Serving. You can take a look at the script, its flags/options, and these resources for more information:
   * [SavedModel](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model)
   * [SignatureDefs](https://www.tensorflow.org/tfx/serving/signature_defs) 
   
6. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`. 
   For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, `num_physical_cores = 28 * 2 = 56`. 
   To compute *num_physical_cores* with bash commands:
   ```
   cores_per_socket=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`
   num_sockets=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
   num_physical_cores=$((cores_per_socket * num_sockets))
   echo $num_physical_cores
   ```

7. **Start the server**: Now let's start up the TensorFlow model server. To optimize overall performance, use the following recommended settings from the
   [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md):
   * OMP_NUM_THREADS=*num_physical_cores*
   * TENSORFLOW_INTER_OP_PARALLELISM=2
   * TENSORFLOW_INTRA_OP_PARALLELISM=*num_physical_cores*
   
   For our example with 56 physical cores:
   ```
   docker run \
           --name=tfserving \
           --rm \
           -d \
           -p 8500:8500 \
           -v "/tmp:/models/inceptionv3" \
           -e MODEL_NAME=inceptionv3 \
           -e OMP_NUM_THREADS=$num_physical_cores \
           -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
           -e TENSORFLOW_INTRA_OP_PARALLELISM=$num_physical_cores \
           intel/intel-optimized-tensorflow-serving:2.3.0
   ```
   **Note:** For some models, playing around with these settings values can improve performance even further. 
   We recommend that you experiment with your own hardware and model if you have strict performance requirements.

8. **Run a Test**: Now we can run a test client that downloads a cat picture and sends it for recognition.
   The script has an option for sending a local JPG, if you would prefer to test a different image.
   Run `python image_recognition_client.py --help` for more usage information.
   ```
   python image_recognition_client.py --model inceptionv3
   ```
   The output should say `Predicted class:  286`.
   
   **Note**: After running some basic tests, you may wish to constrain the inference server to a single socket. 
   Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources.
   * See our [Best Practices document](/docs/general/tensorflow_serving/GeneralBestPractices.md#docker-cpuset-settings) for information and examples
   * See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for more options and definitions
   
9. **Online inference**: Online (or real-time) inference is usually defined as the time it takes to return a prediction for batch size 1.
   To see average online inference performance (in ms), run the script `image_recognition_benchmark.py` using batch_size 1:
   ```
   python image_recognition_benchmark.py --batch_size 1 --model inceptionv3
   ```
   
   Console out:
   ```
   Iteration 1: ... sec
   ...
   Iteration 40: ... sec
   Average time: ... sec
   Batch size = 1
   Latency: ... ms
   Throughput: ... images/sec
   ```
   
10. **Batch inference**: Regardless of hardware, the best batch size is 128. 
    To see average batch inference performance (in images/sec), run the script `image_recognition_benchmark.py` using batch_size 128:
    ```
    python image_recognition_benchmark.py --batch_size 128 --model inceptionv3
    ```
    
    Console out:
    ```
    Iteration 1: ... sec
    ...
    Iteration 40: ... sec
    Average time: ... sec
    Batch size = 128
    Throughput: ... images/sec
    ```

11. **Clean up**: 
    * After you are finished sending requests to the server, you can stop the container running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
	  
	  ```
	  docker rm -f tfserving
	  ```
    
    * Deactivate your virtual environment with `deactivate`.
    
## Conclusion

You have now seen three end-to-end examples of serving an image recognition model for inference using TensorFlow Serving, and learned:
1. How to create a SavedModel from a TensorFlow model graph
2. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
3. How to verify that the served model can correctly classify an image using a gRPC client
4. How to measure online and batch inference metrics using a gRPC client

With this knowledge and the example code provided, 
you should be able to get started serving your own custom image recognition model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement are possible.
