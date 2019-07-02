
# Recommendation System with TensorFlow Serving on CPU using NCF

## Goal

This tutorial will introduce you to the CPU performance considerations for recommendation systems and how to use [Intel® Optimizations for TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPUs. 
This tutorial uses a pre-trained [Neural Collaborative Filtering (NCF)](https://arxiv.org/pdf/1708.05031.pdf) model for recommending movies to users. 
We provide sample code that you can use to get your optimized TensorFlow model server and GRPC client up and running quickly. 
In this tutorial using NCF, you will measure inference performance in two situations:
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

Neural Collaborative Filtering (NCF) is a deep learning architecture for personalized recommendation systems. 
NCF uses a neural network to model user-item interaction instead of the traditional approach of matrix factorization using an inner product over latent features of users and items. 
In this tutorial, we use pre-trained weights for the NeuMF variant of NCF from the [TensorFlow models repository](https://github.com/tensorflow/models/tree/master/official/recommendation) that predicts users' movie preference based on their past movie ratings.
The model was trained using the MovieLens (1M) dataset, and we will use an evaluation subset from this same dataset for inference through TensorFlow Serving.
For more detailed descriptions of the model topology and MovieLens dataset, please refer to the [TensorFlow models NCF README](https://github.com/tensorflow/models/tree/master/official/recommendation). 

[Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)](https://github.com/intel/mkl-dnn) offers significant performance improvements for many neural network operations. 
Tuning TensorFlow Serving to take full advantage of your hardware for recommendation systems inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a GRPC client to verify prediction accuracy and measure online and batch inference performance
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial with pre-trained NCF model

1. **Download the pre-trained model checkpoints**: Download and extract the packaged pre-trained model checkpoints `ncf_fp32_pretrained_model.tar.gz`
   (refer to the [model README](/benchmarks/recommendation/tensorflow/ncf) to get the latest location of this archive).

   ```
   cd ~
   mkdir ncf
   cd ncf
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ncf_fp32_pretrained_model.tar.gz
   tar -xzvf ncf_fp32_pretrained_model.tar.gz
   ```

   After extraction, you should see a folder named `ncf_trained_movielens_1m`. Let's set write access on this folder because we will need it for the tutorial.
   ```
   chmod -R u+w ncf_trained_movielens_1m
   ```
   
2. **Clone this repository**: Clone the [IntelAI/models](https://github.com/intelai/models) repository into your home directory.

   ```
   cd ~
   git clone https://github.com/IntelAI/models.git
   ```
   
3. **Clone the TensorFlow/models repository**: This tutorial uses code in the `v1.11` tag of the TensorFlow/models repository.
   We also have to make a small modification to `data_async_generation.py` in order to use the model code.

   ```
   mkdir tensorflow-models
   cd tensorflow-models
   git clone https://github.com/tensorflow/models.git
   cd models
   git checkout v1.11
   sed -i.bak 's/atexit.register/# atexit.register/g' official/recommendation/data_async_generation.py
   ```
   
   Now add the repository directory to the `PYTHONPATH` variable:
   
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

4. **Set up a virtual environment**: We will use a virtual environment to install the required packages. 
	
   - If you do not have pip or virtualenv, you will need to get them first:
	 
	 ```
	 sudo apt-get install -y python python-pip virtualenv
	 ```
		
   - Create and activate the python virtual environment in your home directory and install the `tensorflow-serving-api` package.
   
     ```
     cd ~
     virtualenv ncf_venv
     source ncf_venv/bin/activate
     pip install tensorflow-serving-api
     ```
     
   - In addition to the requirements for the GRPC client, we need some additional libraries in order to build, export, and evaluate the model.
     These come in a requirements file in the TensorFlow/models repo.
   
     ```
     pip install -r tensorflow-models/models/official/requirements.txt
     ```

5. **Prepare the data directory**: This tutorial uses the [MovieLens (1M)](https://grouplens.org/datasets/movielens/1m/) dataset. 
   It will automatically be downloaded by the script in step 6. 
   If the `--data_dir` flag is not set, a folder will be created in `/tmp`, but if you prefer to download the data to your home directory and provide the path to `--data_dir`, create an empty directory called `movielens`:

   ```
   mkdir movielens
   ```

6. **Create a SavedModel**: We will use the Model Zoo's [`ncf_main.py`](/models/recommendation/tensorflow/ncf/inference/fp32/ncf_main.py) script to convert the pre-trained model checkpoints to a SavedModel.
   It sets up a TensorFlow Estimator object, defines a `serving_input_receiver_fn()` function, and exports the model in a format compatible with TensorFlow Serving.
   
   ```
   cd ~
   python models/models/recommendation/tensorflow/ncf/inference/fp32/ncf_main.py \
       --data_dir=$HOME/movielens \
       --model_dir=$HOME/ncf/ncf_trained_movielens_1m \
       --inference_only \
       --export_savedmodel
   ```
   
   This script will first test NCF inference in (non-serving) TensorFlow and then create a `~/ncf/ncf_trained_movielens_1m/<timestamp>` directory with a `saved_model.pb` file in it. 
   This is the file we will serve from TensorFlow Serving. Note that if you run the script multiple times with the `--export_savedmodel` flag, it will create multiple `<timestamp>` folders.
   TensorFlow Serving will always use the latest one.
   
   For more information, refer to these resources:
   * [SavedModel](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model)
   * [Using SavedModel with Estimators](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)
   
7. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`. 
   For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, `num_physical_cores = 28 * 2 = 56`. 
   To compute *num_physical_cores* with bash commands:
   ```
   cores_per_socket=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`
   num_sockets=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
   num_physical_cores=$((cores_per_socket * num_sockets))
   echo $num_physical_cores
   ```

8. **Recommended Settings**: To optimize overall performance, you can start with the below settings or those from the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md) and tune from there. 
   We have found empirically that optimal settings for NCF are different depending on the size of the batch. 
   Playing around with the settings can improve performance further, so you should experiment with your own hardware and model if you have strict performance requirements.
   
   Recommendations:
   
   | Options  | Large Batch (50,000-500,000)| Online / Small Batch (1-500) |
   | ------------- | ------------- | ------------- |
   |TENSORFLOW_INTER_OP_PARALLELISM | 1 | 1 |
   |TENSORFLOW_INTRA_OP_PARALLELISM| $num_physical_cores| 1 |
   |OMP_NUM_THREADS |$num_physical_cores| 1 |
      
   **Note**: We are exploring the performance trade-offs for NCF and different metrics for better evaluating performance in TensorFlow Serving.
   
9. **Start the server**: We can now start up the TensorFlow model server. Using `-d` (for "detached") runs the container as a background process.
 
   ```
   cd ~
   docker run \
        --name=tfserving \
        -d \
        -p 8500:8500 \
        -v "$HOME/ncf/ncf_trained_movielens_1m:/models/ncf" \
        -e MODEL_NAME=ncf \
        -e OMP_NUM_THREADS=$num_physical_cores \
        -e TENSORFLOW_INTER_OP_PARALLELISM=1 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=$num_physical_cores \
        tensorflow/serving:mkl
   ```
   
   You can make sure the container is running using the `docker ps` command.

10. **Online and batch performance**: Run `run_ncf.py` [python script](/docs/recommendation/tensorflow_serving/run_ncf.py), which can measure both online and batch performance. 
   
    Go to the tutorial directory:
    ```
    cd ~/models/docs/recommendation/tensorflow_serving
    ```
   
    **Online Inference** (batch_size=1):
    ```
    python run_ncf.py \
        -d $HOME/movielens \
        -b 1 \
        -n 40
    ```
   
    **Batch Inference**:
    ```
    python run_ncf.py \
        -d $HOME/movielens \
        -b 50000 \
        -n 40
    ```
   
    **Note**: The `-n` argument stands for `num_iterations` and if *not* set, the script will run through the entire validation dataset,
    which can take some time, especially for batch size 1. Using 40 iterations will quickly give a good indication of performance. 

11. **Verify accuracy**: To ensure the NCF SavedModel is producing the expected results, leave `num_iterations` at its default value and run the script with a medium or large batch size:
    ```
    python run_ncf.py \
        -d $HOME/movielens \
        -b 50000
    ```
    
    This will send as many batches as necessary to perform inference on the whole evaluation set and then calculate the accuracy metrics **Hit Ratio** (HR) and **Normalized Discounted Cumulative Gain** (NDCG).
    These are non-deterministic, but HR should be greater than 0.22 and NDCG should be greater than 0.11. For more information about the calculation of these metrics, see [`ncf_main.py`](/models/recommendation/tensorflow/ncf/inference/fp32/ncf_main.py#L108).

    Example output:
    ```
    ...
    Accuracy: 0.22285 HR, 0.11215 NDCG
    ```

12. **Clean up**: 
    * After you are finished sending requests to the server, you can stop the container running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
	  
	  ```
	  docker rm -f tfserving
	  ```
    
    * Deactivate your virtual environment with `deactivate`.
    

## Conclusion
You have now seen an end-to-end example of serving a NCF recommendation system model for inference using TensorFlow Serving, and learned:
1. How to export a SavedModel from a TensorFlow Estimator with checkpoints
2. How to choose good starting values for the performance-related runtime parameters exposed by the `docker run` command
3. How to test online and batch inference metrics using a GRPC client
4. How to verify NCF model accuracy by computing HR and NDCG over the whole evaluation dataset

With this knowledge and the example code provided, you should be able to get started serving your own custom recommendation system model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvements are possible for your use case.
