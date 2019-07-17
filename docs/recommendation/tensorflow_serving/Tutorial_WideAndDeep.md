
# Recommendation System with TensorFlow Serving on CPU using Wide and Deep

## Goal

This tutorial will introduce you to the CPU performance considerations for recommendation systems and how to use [Intel® Optimizations for TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPUs. 
This tutorial uses a pre-trained [Wide and Deep](https://arxiv.org/abs/1606.07792) model for predicting advertisement click-throughs, using a dataset from [Kaggle's Display Advertising Challenge](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/). 
We provide sample code that you can use to get your optimized TensorFlow model server and GRPC client up and running quickly. 
In this tutorial using Wide and Deep, you will measure inference performance in two situations:
* **Online inference**, where batch_size=1. In this case, a lower number means better runtime performance.
* **Batch inference**, where batch_size>1. In this case, a higher number means better runtime performance.

**NOTE about GRPC vs. REST**: It [has been suggested](https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62) that GRPC has faster client-side serialization and de-serialization than REST, especially if you are optimizing for batch inference. 
Please note however that this tutorial is focused on optimizing the model server, not the client that sends requests. 
We use GRPC in this tutorial for illustration, not as a best practice, and offer another [tutorial](/docs/object_detection/tensorflow_serving/Tutorial.md) that illustrates the use of the REST API with TensorFlow Serving, if you are interested in that protocol. 

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * [Performance Metrics](/docs/general/tensorflow_serving/GeneralBestPractices.md#performance-metrics)
   * [TensorFlow Serving Configuration Settings](/docs/general/tensorflow_serving/GeneralBestPractices.md#tensorflow-serving-configuration-settings)
* Ran an example end-to-end using a GRPC client, such as the [one in the Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md#option-2-query-using-grpc)
  
## Background

The Wide and Deep model topology combines both a linear model and a deep neural network to exploit the strengths of both. 
The linear model uses a large set of derived features and is good at memorizing relationships between features and outputs. 
This is considered the "wide" component of Wide and Deep. The "deep" component is a deep neural network that uses lower-dimensional dense representation features called embeddings. 
The deep neural network is better at generalization and can identify a more diverse set of inputs with similar outputs, such as individuals who might click an ad.

[Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)](https://github.com/intel/mkl-dnn) offers significant performance improvements for many neural network operations. 
Tuning TensorFlow Serving to take full advantage of your hardware for recommendation systems inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a GRPC client to verify prediction accuracy and measure online and batch inference performance
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial with pre-trained Wide and Deep model

1. **Clone this repository**: Clone the [intelai/models](https://github.com/intelai/models) repository into your home directory.

   ```
   cd ~
   git clone https://github.com/IntelAI/models.git
   ```
   
2. **Prepare the data**: Follow the instructions below to download and prepare the dataset. 
   - Navigate to the models directory of the Model Zoo:

     ```
     cd ~/models/models
     ```
	
   - Download the evaluation data set from Criteo: 

     ```wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv```
  
   - Start a Docker container for preprocessing. You should already have Docker installed and be well-practiced from the [prerequisites](#prerequisites).

     ```
     docker run -it --privileged -u root:root \
             -w /models \
             --volume $PWD:/models \
             docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
             /bin/bash
     ```
    
   - Preprocess and convert the eval dataset to TFRecord format. We will use a script in the Intel Model Zoo repository.
     This step may take a while to complete. You can ignore compiler warnings in the script output. 
   
     ```
     python recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
         --csv-datafile eval.csv \
         --outputfile-name preprocessed
     ```
   
   - Exit the docker container and find the preprocessed dataset `eval_preprocessed.tfrecords` in the location `~/models/models`.

3. **Set up the client environment**: We need to create a virtual environment for this tutorial.
	
   - We will use a virtual environment to install the required packages. If you do not have pip or virtualenv, you will need to get them first:
	 
	 ```
	 sudo apt-get install -y python python-pip
	 pip install virtualenv
	 ```
		
   - Create and activate the python virtual environment in your home directory and install the `tensorflow-serving-api` package.
   
     ```
     cd ~
     virtualenv wd_venv
     source wd_venv/bin/activate
     pip install tensorflow-serving-api
     ```
   
4. **Download the pre-trained model**: Download the pre-trained model `wide_deep_fp32_pretrained_model.pb` into this tutorial's location
   (refer to the [Wide and Deep README](/benchmarks/recommendation/tensorflow/wide_deep_large_ds) to get the latest location of the pre-trained model).

   ```
   cd ~/models/docs/recommendation/tensorflow_serving
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wide_deep_fp32_pretrained_model.pb
   ```

5. **Create a SavedModel**: Using the conversion script `wide_deep_graph_to_saved_model.py`, convert the pre-trained model graph to a SavedModel.
   
   ```
   python wide_deep_graph_to_saved_model.py --import_path wide_deep_fp32_pretrained_model.pb
   ```
   This will export a `saved_model.pb` file to a newly created `/tmp/1/` directory, by default. 
   If you want to export to a location other than `/tmp`, use the `--export_dir` argument.
   If you want to use a different version number than `/1`, use the `--model_version` argument. Just be aware that when you serve the model from TensorFlow Serving, it must be in an integer-named directory (e.g. `1`, `2`, `43858606`, etc.).
   The `saved_model.pb` file has a specific signature definition to make it compatible with TensorFlow Serving.
   You can take a look at the [`wide_deep_graph_to_saved_model.py`](wide_deep_graph_to_saved_model.py) script, its flags/options, and these resources for more information:
   * [SavedModel](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model)
   * [SignatureDefs](https://www.tensorflow.org/serving/signature_defs) 
   
6. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`. 
   For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, `num_physical_cores = 28 * 2 = 56`. 
   To compute *num_physical_cores* with bash commands:
   
   ```
   cores_per_socket=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`
   num_sockets=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
   num_physical_cores=$((cores_per_socket * num_sockets))
   echo $num_physical_cores
   ```

7. **Recommended Settings**: There is no one-size-fits-all solution to optimizing Wide and Deep performance on CPUs, but understanding the bottlenecks and tuning the run-time parameters based on your dataset and TensorFlow graph can be helpful.
   The "wide" linear component does not provide opportunities for parallelism within each node, but the "deep" component does benefit from more parallelism within each node.
   The best choice for the parameters depends on the number of features in your dataset and number of hidden units in the deep model.
   
   To optimize performance, you can start with the below settings or those from the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md) and tune from there.
   For Wide and Deep, we have found that Intel MKL-DNN offers greater performance improvements over the default TensorFlow Serving installation the larger the batch size.
   For very small batch sizes (i.e. 1-32), general best settings are especially hard to identify, so we recommend that you experiment with your own hardware and model.
   
   | Options  | Large Batch (>=128) | Online / Small Batch (<128) |
   | ------------- | ------------- | ------------- |
   | TENSORFLOW_INTER_OP_PARALLELISM | 1 | $num_physical_cores |
   | TENSORFLOW_INTRA_OP_PARALLELISM | 1 | $num_physical_cores |
   | OMP_NUM_THREADS | $num_physical_cores | $num_physical_cores/2 |
   
8. **Start the server**: We can now start up the TensorFlow model server. Using `-d` (for "detached") runs the container as a background process.
   If the `saved_model.pb` file was not exported to the `/tmp` directory in step 5, you will need to change the `-v` input below to reflect the custom `--export_dir` (i.e. `-v "/<export_dir>:/models/wide_deep"`).
 
   ```
   docker run \
        --name=tfserving \
        -d \
        -p 8500:8500 \
        -v "/tmp:/models/wide_deep" \
        -e MODEL_NAME=wide_deep \
        -e OMP_NUM_THREADS=$num_physical_cores \
        -e TENSORFLOW_INTER_OP_PARALLELISM=1 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=1 \
        tensorflow/serving:mkl
   ```
   
   You can make sure the container is running using the `docker ps` command.

9. **Online and Batch Inference Performance**: Run `run_wide_deep.py` [python script](/docs/recommendation/tensorflow_serving/run_wide_deep.py), which can measure both online and batch inference performance. 
   
   **Online Inference** (batch_size=1):
   ```
   python run_wide_deep.py -d ~/models/models/eval_preprocessed.tfrecords -b 1
   ```
   
   **Batch Inference** (e.g. batch_size=512):
   ```
   python run_wide_deep.py -d ~/models/models/eval_preprocessed.tfrecords -b 512
   ```

10. **Clean up**: 
    * After you are finished sending requests to the server, you can stop the container running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
	  
	  ```
	  docker rm -f tfserving
	  ```
    
    * Deactivate your virtual environment with `deactivate`.
    

## Conclusion
You have now seen an end-to-end example of serving a recommendation system model for inference using TensorFlow Serving, and learned:
1. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
2. How to test online and batch inference metrics using a GRPC client

With this knowledge and the example code provided, you should be able to get started serving your own custom recommendation model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement are possible.
