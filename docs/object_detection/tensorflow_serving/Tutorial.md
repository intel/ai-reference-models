
# Object Detection with TensorFlow Serving on CPU using R-FCN model

## Goal

This tutorial will introduce you to the CPU performance considerations for object detection in deep learning models and how to use [Intel® Optimizations for TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPUs. 
This tutorial uses a pre-trained Region-based Fully Convolutional Network (R-FCN) model for object detection and provides sample code that you can use to get your optimized TensorFlow model server and REST client up and running quickly. In this tutorial using R-FCN, you will measure inference performance in two situations:
* **Real-Time**, where batch_size=1. In this case, lower latency means better runtime performance.
* **Throughput**, where batch_size>1. In this case, higher throughput means better runtime performance.

**NOTE about REST vs. GRPC**: This tutorial is focused on optimizing the model server, not the client that sends requests. For optimal client-side serialization and de-serialization, you may want to use TensorFlow Serving's GRPC option instead of the REST API, especially if you are optimizing for maximum throughput (here is one [article](https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62) with a relevant analysis). 
We use REST in this tutorial for illustration, not as a best practice, and offer another [tutorial](/docs/image_recognition/tensorflow_serving/Tutorial.md) that illustrates the use of GRPC with TensorFlow Serving. 

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * [Performance Metrics](/docs/general/tensorflow_serving/GeneralBestPractices.md#performance-metrics)
   * [TensorFlow Serving Configuration Settings](/docs/general/tensorflow_serving/GeneralBestPractices.md#tensorflow-serving-configuration-settings)
* Ran an example end-to-end using a REST client, such as the one in the [Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md)
  
## Background

[Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)](https://github.com/intel/mkl-dnn) offers significant performance improvements for convolution, pooling, normalization, activation, and other operations for object detection, using efficient vectorization and multi-threading. Tuning TensorFlow Serving to take full advantage of your hardware for object detection deep learning inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a REST client notebook to verify object detection and measure latency and throughput
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial with pre-trained R-FCN model

1. **Set up your environment**: We need to setup two things for this tutorial
	#### 1.1 Install the [requests](http://docs.python-requests.org) package for making REST HTTP requests. 
	We will use a virtual environment to install the required packages. If you do not have pip or virtualenv, you will need to get them first:
	```
	$ sudo apt-get install -y python python-pip
	$ pip install virtualenv
	```
		
	Create and activate the python virtual envirnoment in your home directory and install the [`requests`](http://docs.python-requests.org) package.
   ```
   $ cd ~
   $ virtualenv rfcn_venv
   $ source rfcn_venv/bin/activate
   (rfcn_venv)$ pip install requests
   ```
   
	#### 1.2 Install [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
	 For detailed instructions, [click here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). Following are the instructions for Ubuntu 16.04.
	
	
	1.2.1 Install Tensorflow Object Detection API dependencies
	```
	(rfcn_venv)$ sudo apt-get install -y protobuf-compiler python-pil python-lxml python-tk
	(rfcn_venv)$ pip install tensorflow Cython contextlib2 jupyter matplotlib pillow lxml
	```

	1.2.2 Clone the tensorflow models repo into your home directory.
	```
	(rfcn_venv)$ cd ~
	(rfcn_venv)$ git clone https://github.com/tensorflow/models
	(rfcn_venv)$ export TF_MODELS_ROOT=$(pwd)/models
	(rfcn_venv)$ echo "export TF_MODELS_ROOT=$(pwd)/models" >> ~/.bashrc
	```

	1.2.3 Install COCO API
	```
	(rfcn_venv)$ cd ~
	(rfcn_venv)$ git clone https://github.com/cocodataset/cocoapi.git
	(rfcn_venv)$ cd cocoapi/PythonAPI
	(rfcn_venv)$ make
	(rfcn_venv)$ cp -r pycocotools $TF_MODELS_ROOT/research/
	```

	1.2.4 Manually install the protobuf-compiler v3.0.0, run the compilation process, add Libraries to PYTHONPATH and to your `.bashrc` and test the installation of Tensorflow Object Detection API
	```
	(rfcn_venv)$ cd $TF_MODELS_ROOT/research/
	(rfcn_venv)$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
	(rfcn_venv)$ unzip protobuf.zip
	(rfcn_venv)$ ./bin/protoc object_detection/protos/*.proto --python_out=.
	(rfcn_venv)$ export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/slim
	(rfcn_venv)$ echo "export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/slim" >> ~/.bashrc
	(rfcn_venv)$ python object_detection/builders/model_builder_test.py
	```
	     
2. **Download the Data**: Download the 2017 validation COCO dataset (~780MB) (**note**: do not convert the COCO dataset to TF records format):
   
   ```
   (rfcn_venv)$ cd ~
   (rfcn_venv)$ mkdir -p coco/val
   (rfcn_venv)$ wget http://images.cocodataset.org/zips/val2017.zip
   (rfcn_venv)$ unzip val2017.zip -d coco/val
   (rfcn_venv)$ export COCO_VAL_DATA=$(pwd)/coco/val/val2017
   (rfcn_venv)$ echo "export COCO_VAL_DATA=$(pwd)/coco/val/val2017" >> ~/.bashrc
   ```
   
3. **Download and Prepare the pre-trained SavedModel**: Download and extract the pre-trained model and copy the `rfcn_resnet101_fp32_coco/saved_model/saved_model.pb` to `rfcn/1` (the `1` subdirectory is important - don't skip it!). This is the file we will serve from TensorFlow Serving.
   Refer to the [TensorFlow documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model) for more information about SavedModels, and refer to this [README file](/benchmarks/object_detection/tensorflow/rfcn/README.md#download_fp32_pretrained_model) to get the latest location of the pre-trained model.
   ```
   (rfcn_venv)$ cd ~/
   (rfcn_venv)$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
   (rfcn_venv)$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
   (rfcn_venv)$ mkdir -p rfcn/1
   (rfcn_venv)$ cp rfcn_resnet101_fp32_coco/saved_model/saved_model.pb rfcn/1
   ```
   
4. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`. For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, `num_physical_cores = 28 * 2 = 56`. To compute *num_physical_cores* and *tf_session_parallelism* with bash commands:
   ```
   (rfcn_venv)$ cores_per_socket=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`
   (rfcn_venv)$ num_sockets=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
   (rfcn_venv)$ num_physical_cores=$((cores_per_socket * num_sockets))
   (rfcn_venv)$ echo $num_physical_cores
   ```

5. **Start the server**: Now let's start up the TensorFlow model server. With `&` at the end of the cmd, runs the container as a background process. Press enter after executing the following cmd. 
To optimize overall performance, use the following recommended settings from the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md):
   * OMP_NUM_THREADS=*num_physical_cores*
   * TENSORFLOW_INTER_OP_PARALLELISM=2
   * TENSORFLOW_INTRA_OP_PARALLELISM=*num_physical_cores*
 
   ```
   (rfcn_venv)$ cd ~
   (rfcn_venv)$ docker run \
        --name=tfserving_rfcn \
        -p 8501:8501 \
        -v "$(pwd)/rfcn:/models/rfcn" \
        -e MODEL_NAME=rfcn \
        -e OMP_NUM_THREADS=$num_physical_cores \
        -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=$num_physical_cores \
        tensorflow/serving:mkl &
   ```
   **Note**: For some models, playing around with these settings values can improve performance even further. 
   We recommend that you experiment with your own hardware and model if you have strict performance requirements.

6. *Measure Real-Time and Throughput performance**: Clone the Intel Model Zoo into a directory called `intel-models` and run `rfcn-benchmark.py` [python script](/docs/object_detection/tensorflow_serving/rfcn-benchmark.py), which will test both Real-Time and Throughput performance. 
      ```
   (rfcn_venv)$ git clone https://github.com/IntelAI/models.git intel-models
   (rfcn_venv)$ python intel-models/docs/object_detection/tensorflow_serving/rfcn-benchmark.py \
     -i $COCO_VAL_DATA
   ```


7. **Visualize Object Detection Output**: To visually see the output of object detection results, we will use Jupyter notebook via web browser. If you are using a system that does not have a browser,  such as a VM on GCP or AWS, a workaround is to use local port forwarding of port 8888 to relay the jupyter service to your localhost. You will need to quit your SSH session and log back in with port forwarding configured. 
For example, with a GCP VM, add `--ssh-flag="-L 8888:localhost:8888"` to your ssh command. Once you are connected again with port forwarding, reactivate the virtual environment, navigate to the tutorial directory, and start jupyter notebook. Continue with the next instruction.
      ```
   $ cd ~
   $ source rfcn_venv/bin/activate
   (rfcn_venv)$ cd intel-models/docs/object_detection/tensorflow_serving
   (rfcn_venv)$ jupyter notebook
   ```
	After running `jupyter notebook` , paste the generated link into your browser and open the `RFCN.ipynb` file. You will need to edit the code in one place - in the second cell, insert the path to your downloaded COCO validation data set. Then, execute the cells in order. The output of the "Test Object Detection" section should be an image with objects correctly detected by the R-FCN model.

8. (Optional) **Using a single core**: In some cases, it is desirable to constrain the inference server to a single core or socket. Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources. See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for all the options and their definitions. For example, to run the container so that a single CPU is used, you can use these settings:
   * `--cpuset-cpus="0"`
   * `--cpus="1"`
   * `OMP_NUM_THREADS=1`
   * `TENSORFLOW_INTER_OP_PARALLELISM=1`
   * `TENSORFLOW_INTRA_OP_PARALLELISM=1`
   
   ```
   (rfcn_venv)$ docker run \
        --name=tfserving_rfcn_1 \
        -p 8500:8500 \
        --cpuset-cpus="0" \
        --cpus="1" \
        -v "$(pwd)/rfcn:/models/rfcn" \
        -e MODEL_NAME=rfcn \
        -e OMP_NUM_THREADS=1 \
        -e TENSORFLOW_INTER_OP_PARALLELISM=1 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=1 \
        tensorflow/serving:mkl &
   ```

10. **Clean up**: 
    * After saving any changes you made to the Jupyter notebook, close the file and stop the Jupyter server by clicking `Quit` from the main file browser. 
    * After you are fininshed with querying, you can stop the container which is running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
		```
		 (rfcn_venv)$ docker rm -f tfserving_rfcn
		```
    * Deactivate your virtual environment with `deactivate`.
    

## Conclusion
You have now seen an end-to-end example of serving an object detection model for inference using TensorFlow Serving, and learned:
1. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
2. How to verify that the served model can correctly detect objects in an image using a sample Jupyter notebook
3. How to measure latency and throughput metrics using a REST client

With this knowledge and the example code provided, you should be able to get started serving your own custom object detection model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement are possible.
