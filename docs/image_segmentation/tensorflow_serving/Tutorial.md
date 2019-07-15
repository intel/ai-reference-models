
# Image Segmentation with TensorFlow Serving on CPU with Mask R-CNN

## Goal

This tutorial will introduce you to the CPU performance considerations for image segmentation in deep learning models and how to use [Intel® Optimizations for TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPUs. 
This tutorial uses a pre-trained Mask R-CNN model for image segmentation and provides sample code that you can use to get your optimized TensorFlow model server and client up and running quickly.
In this tutorial you will measure **online inference performance**, where batch_size=1 and a lower number means better runtime performance.

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * [Performance Metrics](/docs/general/tensorflow_serving/GeneralBestPractices.md#performance-metrics)
   * [TensorFlow Serving Configuration Settings](/docs/general/tensorflow_serving/GeneralBestPractices.md#tensorflow-serving-configuration-settings)
* Ran an example end-to-end using a GRPC client, such as the one in the [Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md)
  
## Background

[Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)](https://github.com/intel/mkl-dnn) offers significant performance improvements for convolution, pooling, normalization, activation, and other operations for image segmentation, using efficient vectorization and multi-threading.
Tuning TensorFlow Serving to take full advantage of your hardware for image segmentation deep learning inference involves:
1. Running a TensorFlow Serving docker container configured for performance given your hardware resources
2. Running a GRPC client to verify image segmentation results and measure online inference performance
3. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial

1. **Download the data and clone the Model Zoo**: 

   1.1 Download the 2014 validation COCO dataset and minival annotations:   
   ```
   cd ~
   mkdir coco
   cd coco
   wget http://images.cocodataset.org/zips/val2014.zip
   unzip -q val2014.zip
   wget http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/coco_minival2014.tar.gz
   mkdir annotations
   tar -xzf coco_minival2014.tar.gz -C annotations
   ```
   
   1.2 Clone the Intel Model Zoo into your home directory and add the Mask R-CNN model folder to your PYTHONPATH:
   ```
   cd ~
   git clone https://github.com/IntelAI/models.git
   export PYTHONPATH=$PYTHONPATH:$(pwd)/models/models/image_segmentation/tensorflow/maskrcnn
   ```

2. **Clone the Mask R-CNN repository**: Clone the Mask R-CNN model repository. It is used as an external source for dependencies.
   ```
   cd ~
   git clone https://github.com/matterport/Mask_RCNN.git
   ```

3. **Set up your Python3 virtual environment**: We will use a virtual environment to install the required packages.
   Mask R-CNN requires Python3.
   
   3.1 Install python3-venv, python3-dev, and python3-pip and upgrade to the latest pip3 version:
   ```
   cd ~
   sudo apt-get install python3-venv python3-dev python3-pip
   pip3 install --upgrade pip
   
   ```
   
   3.2 Create a new Python3 virtual environment, activate it, upgrade pip once more, and upgrade setuptools:
   ```
   python3 -m venv mask_env
   source mask_env/bin/activate
   pip install --upgrade pip
   pip3 install --upgrade setuptools
   ```
   
   3.3 Install requirements from the Mask R-CNN repository:
   ```
   pip3 install -r Mask_RCNN/requirements.txt
   ```
   
   3.4 Install remaining dependencies for the tutorial:
   ```
   pip3 install jupyter tensorflow-serving-api
   ```
   
   3.5 Downgrade to scipy version 1.2.1 due to discontinued functions needed by the model. 
   ```
   pip3 install --force-reinstall scipy==1.2.1
   ```
   
4. **Download the pre-trained model weights**: Download the `.h5` file from the Mask R-CNN repository release page, and place it in the Mask_RCNN directory .
   ```
   cd ~/Mask_RCNN
   wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
   ```

5. **Create a SavedModel**: Using the conversion script `mask_rcnn_weights_to_saved_model.py`, convert the pre-trained model weights to a SavedModel.
   
   ```
   cd ~/models/docs/image_segmentation/tensorflow_serving
   python mask_rcnn_weights_to_saved_model.py --import_path ~/Mask_RCNN/mask_rcnn_coco.h5
   ```
   
   This will create a `/tmp/1/` directory with a `saved_model.pb` file in it. This is the file we will serve from TensorFlow Serving.
   The [`mask_rcnn_weights_to_saved_model.py`](mask_rcnn_weights_to_saved_model.py) script constructs a frozen graph from the model code and weights and attaches a signature definition to the model in order to make it compatible with TensorFlow Serving.
   You can take a look at the script, its flags/options, and these resources for more information:
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

7. **Start the server**: Now start up the TensorFlow model server. Using `-d` (for "detached") runs the container as a background process. 
   To optimize overall performance, use the following recommended settings from the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md):
     * OMP_NUM_THREADS=*num_physical_cores*
     * TENSORFLOW_INTER_OP_PARALLELISM=2
     * TENSORFLOW_INTRA_OP_PARALLELISM=*num_physical_cores*
 
   ```
   cd ~
   docker run \
       --name=tfserving \
       -d \
       -p 8500:8500 \
       -v "/tmp:/models/mask_rcnn" \
       -e MODEL_NAME=mask_rcnn \
       -e OMP_NUM_THREADS=$num_physical_cores \
       -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
       -e TENSORFLOW_INTRA_OP_PARALLELISM=$num_physical_cores \
       tensorflow/serving:mkl
   ```
   
   **Note**: For some models, playing around with the parallelism settings can improve performance even further. 
   We recommend that you experiment with your own hardware and model if you have strict performance requirements.

8. **Measure online inference performance**: Run the `run_mask_rcnn.py` [python script](/docs/image_segmentation/tensorflow_serving/run_mask_rcnn.py), which will test online inference (batch size 1). 
   
   ```
   cd ~
   python models/docs/image_segmentation/tensorflow_serving/run_mask_rcnn.py \
       -m Mask_RCNN \
       -i coco/val2014
   ```

9. **Visualize image segmentation output**: To visually see the results of image segmentation, we will use a Jupyter notebook via web browser. 
   If you are using a system that does not have a browser,  such as a VM on GCP or AWS, a workaround is to use local port forwarding of port 8888 to relay the jupyter service to your localhost.
   You will need to quit your SSH session and log back in with port forwarding configured. For example, with a GCP VM, add `--ssh-flag="-L 8888:localhost:8888"` to your ssh command. 
   Once you are connected again with port forwarding, reactivate the virtual environment, navigate to the tutorial directory, and start the jupyter notebook service.
   
   ```
   cd ~
   source mask_env/bin/activate
   cd models/docs/image_segmentation/tensorflow_serving
   jupyter notebook
   ```
   
   After running `jupyter notebook`, paste the generated link into your browser and open the `ImageSegmentation.ipynb` file. 
   You will need to edit the code in one cell - in the second cell, insert the full paths to your cloned Mask_RCNN repo, Model Zoo repo, and downloaded COCO validation data set.
   Then, execute the cells in order. The output of the "Test Image Segmentation" section should be an image with objects detected and overlaid with masks.

10. (Optional) **Using a single core**: In some cases, it is desirable to constrain the inference server to a single core or socket. 
    Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources. 
    See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for all the options and their definitions. 
    For example, to run the container so that a single CPU is used, you can use these settings:
    * `--cpuset-cpus="0"`
    * `--cpus="1"`
    * `OMP_NUM_THREADS=1`
    * `TENSORFLOW_INTER_OP_PARALLELISM=1`
    * `TENSORFLOW_INTRA_OP_PARALLELISM=1`
   
    ```
    cd ~
    docker run \
        --name=tfserving_1core \
        -d \
        -p 8500:8500 \
        -p 8501:8501 \
        --cpuset-cpus="0" \
        --cpus="1" \
        -v "/tmp:/models/mask_rcnn" \
        -e MODEL_NAME=mask_rcnn \
        -e OMP_NUM_THREADS=1 \
        -e TENSORFLOW_INTER_OP_PARALLELISM=1 \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=1 \
        tensorflow/serving:mkl
    ```
   
11. **Clean up**: 
    * After saving any changes you made to the Jupyter notebook, close the file and stop the Jupyter server by clicking `Quit` from the main file browser. 
    * After you are finished with querying, you can stop the container which is running in the background. 
    To restart the container with the same name, you need to stop and remove the container from the registry. 
    To view your running containers run `docker ps`.
    
	```
	docker rm -f tfserving
	```
    
    * Deactivate your virtual environment with `deactivate`.
    
## Conclusion
You have now seen an end-to-end example of serving an image segmentation model for inference using TensorFlow Serving, and learned:
1. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
2. How to test online inference performance using a GRPC client
3. How to verify that the served model can correctly detect segments of an image using a sample Jupyter notebook

With this knowledge and the example code provided, you should be able to get started serving your own custom image segmentation model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement is possible.
