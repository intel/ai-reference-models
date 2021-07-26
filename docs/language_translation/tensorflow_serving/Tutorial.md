
# Language Translation with TensorFlow Serving on CPU using Transformer-LT

## Goal

This tutorial will introduce you to the CPU performance considerations for language translation and how to use [Intel® Optimizations for TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) to improve inference time on CPUs. 
This tutorial uses a pre-trained [Transformer-LT](https://arxiv.org/pdf/1706.03762.pdf) model for translating English to German and a sample of English news excerpts from the [WMT14](https://nlp.stanford.edu/projects/nmt/) dataset. 
We provide sample code that you can use to get your optimized TensorFlow model server and gRPC client up and running quickly. 
In this tutorial using Transformer-LT, you will measure inference performance in two situations:
* **Online inference**, where batch_size=1. In this case, a lower number means better runtime performance.
* **Batch inference**, where batch_size>1. In this case, a higher number means better runtime performance.

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
  * [Performance Metrics](/docs/general/tensorflow_serving/GeneralBestPractices.md#performance-metrics)
  * [TensorFlow Serving Configuration Settings](/docs/general/tensorflow_serving/GeneralBestPractices.md#tensorflow-serving-configuration-settings)
* Ran an example end-to-end using a gRPC client, such as the [one in the Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md#option-2-query-using-grpc)

> Note: We use gRPC in this tutorial and offer another [tutorial](/docs/object_detection/tensorflow_serving/Tutorial.md) that illustrates the use of the REST API if you are interested in that protocol. 
  
## Background

The Transformer-LT model is a popular solution for language translation. 
It is based on an encoder-decoder architecture with an added attention mechanism. 
The encoder is used to encode the original sentence to a meaningful fixed-length vector, and the decoder is responsible for extracting the context data from the vector. 
The encoder and decoder process the inputs and outputs, which are in the form of a time sequence.

In a traditional encoder/decoder model, each element in the context vector is treated equally, but this is typically not the ideal solution. 
For instance, when you translate the phrase “I travel by train” from English into Chinese, the word “I” has a greater influence than other words when producing its counterpart in Chinese. 
Thus, the attention mechanism was introduced to differentiate contributions of each element in the source sequence to their counterpart in the destination sequence, through the use of a hidden matrix. 
This matrix contains weights of each element in the source sequence when producing elements in the destination sequence. 

[Intel® oneAPI Deep Neural Network Library (Intel® oneDNN)](https://github.com/oneapi-src/oneDNN) offers significant performance improvements for many neural network operations. 
Tuning TensorFlow Serving to take full advantage of your hardware for language translation inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a gRPC client to verify prediction accuracy and measure online and batch inference performance
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial with pre-trained Transformer-LT (Official) model

1. **Clone this repository**: Clone the [intelai/models](https://github.com/intelai/models) repository into your home directory.

   ```
   cd ~
   git clone https://github.com/IntelAI/models.git
   ```
   
2. **Clone the tensorflow/models repository**: Tokenization of the input data requires utility functions in the tensorflow/models repository.

   ```
   cd ~
   mkdir tensorflow-models
   cd tensorflow-models
   git clone https://github.com/tensorflow/models.git
   cd models
   ```
   
   Now add the required directory to the `PYTHONPATH` variable:
   
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)/official/nlp/transformer
   ```

3. **Set up the client environment**: We need to create a virtual environment for this tutorial.
	
   - We will use a virtual environment to install the required packages. If you do not have pip or virtualenv, you will need to get them first:
	 
	 ```
	 sudo apt-get install -y python python-pip virtualenv
	 ```
		
   - Create and activate the python virtual environment in your home directory and install the `pandas` and `tensorflow-serving-api` packages.
   
     ```
     cd ~
     virtualenv -p python3 lt_venv
     source lt_venv/bin/activate
     pip install pandas tensorflow-serving-api
     ```
   
4. **Download the pre-trained model and test data**: Download and extract the packaged pre-trained model and dataset `transformer_lt_official_fp32_pretrained_model.tar.gz`
   (refer to the [model README](/benchmarks/language_translation/tensorflow/transformer_lt_official) to get the latest location of this archive).

   ```
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/transformer_lt_official_fp32_pretrained_model.tar.gz
   tar -xzvf transformer_lt_official_fp32_pretrained_model.tar.gz
   ```

   After extraction, you should see the following folders and files in the `transformer_lt_official_fp32_pretrained_model` directory:

   ```
   ls -l transformer_lt_official_fp32_pretrained_model/*
   ```
   
   Console out:
   ``` 
   transformer_lt_official_fp32_pretrained_model/data:
   total 1064
   -rw-r--r--. 1 <user> <group> 359898 Feb 20 16:05 newstest2014.en
   -rw-r--r--. 1 <user> <group> 399406 Feb 20 16:05 newstest2014.de
   -rw-r--r--. 1 <user> <group> 324025 Mar 15 17:31 vocab.txt
    
   transformer_lt_official_fp32_pretrained_model/graph:
   total 241540
   -rwx------. 1 <user> <group> 247333269 Mar 15 17:29 fp32_graphdef.pb
   ```

   - `newstest2014.en`: Input file with English text
   - `newstest2014.de`: German translation of the input file for measuring accuracy
   - `vocab.txt`: Dictionary of vocabulary
   - `fp32_graphdef.pb`: Pre-trained model

5. **Create a SavedModel**: Using the conversion script `transformer_graph_to_saved_model.py`, convert the pre-trained model graph to a SavedModel.
   
   ```
   cd ~/models/benchmarks/language_translation/tensorflow_serving/transformer_lt_official/inference/fp32
   python transformer_graph_to_saved_model.py --import_path ~/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb
   ```
   
   This will create a `/tmp/1/` directory with a `saved_model.pb` file in it. This is the file we will serve from TensorFlow Serving.
   The [`transformer_graph_to_saved_model.py`](transformer_graph_to_saved_model.py) script attaches a signature definition to the model in order to make it compatible with TensorFlow Serving.
   You can take a look at the script, its flags/options, and these resources for more information:
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

7. **Recommended Settings**: To optimize overall performance, start with the following settings from the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md).
   Playing around with these settings can improve performance even further, so you should experiment with your own hardware and model if you have strict performance requirements.
   
   | Options  | Recommendations|
   | ------------- | ------------- |
   |TENSORFLOW_INTER_OP_PARALLELISM | 2 |
   |TENSORFLOW_INTRA_OP_PARALLELISM| Number of physical cores | 
   |OMP_NUM_THREADS |Number of physical cores|
   | Batch Size | 64 |
   
8. **Start the server**: We can now start up the TensorFlow model server. Using `-d` (for "detached") runs the container as a background process.
 
   ```
   cd ~
   docker run \
        --name=tfserving \
        -d \
        -p 8500:8500 \
        -v "/tmp:/models/transformer_lt_official" \
        -e MODEL_NAME=transformer_lt_official \
        -e OMP_NUM_THREADS=$num_physical_cores \
        -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=$num_physical_cores \
        intel/intel-optimized-tensorflow-serving:latest
   ```
   
   You can make sure the container is running using the `docker ps` command.
   
   **Note**: After running some basic tests, you may wish to constrain the inference server to a single socket. 
   Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources.
   * See our [Best Practices document](/docs/general/tensorflow_serving/GeneralBestPractices.md#docker-cpuset-settings) for information and examples
   * See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for more options and definitions
  
9. **Online and batch performance**: Run the `transformer_benchmark.py` [python script](/models/benchmarks/language_translation/tensorflow_serving/transformer_lt_official/inference/fp32/transformer_benchmark.py), which can measure both online and batch performance. 
   
   If you are not already there, go to the model's benchmarks directory:
   ```
   cd ~/models/benchmarks/language_translation/tensorflow_serving/transformer_lt_official/inference/fp32
   ```
   
   **Online Inference** (batch_size=1):
   ```
   python transformer_benchmark.py \
       -d ~/transformer_lt_official_fp32_pretrained_model/data/newstest2014.en \
       -v ~/transformer_lt_official_fp32_pretrained_model/data/vocab.txt \
       -b 1
   ```
   
   **Batch Inference** (batch_size=64):
   ```
   python transformer_benchmark.py \
       -d ~/transformer_lt_official_fp32_pretrained_model/data/newstest2014.en \
       -v ~/transformer_lt_official_fp32_pretrained_model/data/vocab.txt \
       -b 64
   ```
   
   **Note**: If you want an output file of translated sentences, set the `-o` flag to an output file name of your choice.
   If this option is set, the script will take a significantly longer time to finish.

10. **Clean up**: 
    * After you are finished sending requests to the server, you can stop the container running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
	  
	  ```
	  docker rm -f tfserving
	  ```
    
    * Deactivate your virtual environment with `deactivate`.
    

## Conclusion
You have now seen an end-to-end example of serving a language translation model for inference using TensorFlow Serving, and learned:
1. How to create a SavedModel from a Transformer-LT TensorFlow model graph
2. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
3. How to test online and batch inference metrics using a gRPC client

With this knowledge and the example code provided, you should be able to get started serving your own custom language translation model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvements are possible.
