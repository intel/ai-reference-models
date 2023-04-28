# Intel® Optimization for TensorFlow Serving Installation (Linux)

## Goal
This tutorial will guide you through step-by-step instructions for
* [Installing Intel® Optimization for TensorFlow Serving as Docker image](#installation)
* Running an example - [serving ResNet-50 v1 saved model using REST API and gRPC](#example-serving-resnet-50-v1-model).

## Prerequisites
1.  Access to a machine with the following resources:
    * **Hardware recommendations**
      * minimum of **20 GB of free disk space** (required) 
      * minimum of **8 logical cores** (highly recommended).
      * We recommend the following Instance types for cloud VM's which have the latest Intel® Xeon® Processors:
        * AWS: [C5 Instances](https://aws.amazon.com/ec2/instance-types/c5/) (Helpful guides: [Get_Started](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html), [Accessing_Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html))
        * GCP: ["Intel Skylake" or "Intel Cascade Lake" CPU Platform](https://cloud.google.com/compute/docs/cpu-platforms) (Helpful guides: [Get_Started](https://cloud.google.com/compute/docs/instances/create-start-instance), [Accessing_Instances](https://cloud.google.com/sdk/gcloud/reference/compute/ssh))
        * Azure: [Fsv2-series](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/#f-series), [Hc-series](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/#hc-series), or [M-series](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/#m-series) (Helpful guides: [Get_Started](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/), [Accessing_Instances](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-cli))

    * **Software recommendations**
      * Ubuntu 16.04 as these instructions were written and tested for it, but the process should be very similar for any other Linux distribution.
      * SSH login and HTTP/S traffic enabled. For details, contact your system administrator or see cloud provider console documentation ([AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html), [GCP](https://cloud.google.com/sdk/gcloud/reference/compute/ssh), [Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-cli)).

2. Install Docker CE
   * Click [here](https://docs.docker.com/engine/install/ubuntu/) for Ubuntu instructions. For other OS platforms, see [here](https://docs.docker.com/get-docker/).
   * Setup docker to be used as a non-root user, to run `docker` commands without `sudo` . Exit and restart your SSH session so that your username is in effect in the docker group.
     ```
     sudo usermod -aG docker `whoami`
     ```
   * After exiting and restarting your SSH session, you should be able to run `docker` commands without `sudo`.
     ```
     docker run hello-world
     ```
     **NOTE**: If your machine is behind a proxy, See **HTTP/HTTPS proxy** section [here](https://docs.docker.com/config/daemon/systemd/)

## Installation
We will break down the installation into 2 steps: 
* Step 1: Pull or build the Intel Optimized TensorFlow Serving Docker image
* Step 2: Verify the Docker image by serving a simple model - half_plus_two

### Step 1: Pull or build TensorFlow Serving Docker image.
The recommended way to use TensorFlow Serving is with Docker images. The easiest way to get an image is to pull the latest version from Docker Hub.
```
docker pull intel/intel-optimized-tensorflow-serving:latest
```

* Login into your machine via SSH and clone the [Tensorflow Serving](https://github.com/tensorflow/serving/) repository and save the path of this cloned directory (Also, adding it to `.bashrc` ) for ease of use for the remainder of this tutorial. 
	```
	git clone https://github.com/ashahba/serving.git // Revert to Upstream once intel-openmp changes are merged
	export TF_SERVING_ROOT=$(pwd)/serving
	echo "export TF_SERVING_ROOT=$(pwd)/serving" >> ~/.bashrc
	```

If you pulled the image and cloned the repository, you can move on to step 2. Alternatively, you can build an image with TensorFlow Serving optimized for Intel® Processors. 

* Using `Dockerfile.devel-mkl`, build an image with Intel optimized ModelServer. This creates an image with all the required development tools and builds from sources. The image size will be around 5GB and will take some time. On AWS c5.4xlarge instance (16 logical cores), it took about 25min.
  
    **NOTE**: It is recommended that you build an official release version using `--build-arg TF_SERVING_VERSION_GIT_BRANCH="<release_number>"`, but if you wish to build the (unstable) head of master, omit the build argument and master will be used by default.
	
	```bash
	export TF_SERVING_VERSION_GIT_BRANCH="ashahba/intel-openmp" // Revert to Upstream once intel-openmp changes are merged
	export TF_SERVING_VERSION_GIT_COMMIT="HEAD"
	export TF_SERVING_BUILD_IMAGE="intel/intel-optimized-tensorflow-serving:${TF_SERVING_VERSION_GIT_BRANCH}"

	cd $TF_SERVING_ROOT/tensorflow_serving/tools/docker/
	git checkout ${TF_SERVING_VERSION_GIT_BRANCH}
	git pull origin

	docker build \
        -f Dockerfile.devel-mkl \
        --build-arg TF_SERVING_VERSION_GIT_BRANCH=${TF_SERVING_VERSION_GIT_BRANCH} \
        --build-arg TF_SERVING_VERSION_GIT_COMMIT=${TF_SERVING_VERSION_GIT_COMMIT} \
        --build-arg TF_SERVING_BUILD_OPTIONS="--config=mkl --config=release" \
        -t ${TF_SERVING_BUILD_IMAGE}-devel .
	```
* Next, using `Dockerfile.mkl`, build a serving image which is a light-weight image without any development tools in it. `Dockerfile.mkl` will build a serving image by copying Intel optimized libraries and ModelServer from the development image built in the previous step:

	```bash
	cd $TF_SERVING_ROOT/tensorflow_serving/tools/docker/
	docker build \
        -f Dockerfile.mkl \
        --build-arg TF_SERVING_BUILD_IMAGE=${TF_SERVING_BUILD_IMAGE}-devel \
	    -t ${TF_SERVING_BUILD_IMAGE} .
	```

	**NOTE 1**: Docker build commands require a `.` path argument at the end; see [docker examples](https://docs.docker.com/engine/reference/commandline/build/#examples) for more background.
		
	**NOTE 2**: If your machine is behind a proxy, you will need to pass proxy arguments to both build commands. For example:
	```
	--build-arg http_proxy="http://proxy.url:proxy_port" --build-arg https_proxy="http://proxy.url:proxy_port"
	```
* Once you built both the images, you should be able to list them using command `docker images`
	```
	docker images
	REPOSITORY                                 TAG                         IMAGE ID            CREATED             SIZE
	intel/intel-optimized-tensorflow-serving   ashahba/intel-openmp        d33c8d849aa3        7 minutes ago       325MB
	intel/intel-optimized-tensorflow-serving   ashahba/intel-openmp-devel  a2e69840d5cc        8 minutes ago       6.58GB
	ubuntu                                     20.04                       20bb25d32758        13 days ago         87.5MB
	hello-world                                latest                      fce289e99eb9        5 weeks ago         1.84kB
	```
	
### Step 2: Verify the Docker image by serving a simple model - half_plus_two

Now let's test the server by serving a simple oneDNN version of half_plus_two model which is included in the repo which we cloned in the previous step.

* Set the location of test model data:
	```
	export TEST_DATA=$TF_SERVING_ROOT/tensorflow_serving/servables/tensorflow/testdata
	```
* Start the container 
    * with `-d`, runs the container as a background process
	* with `-p`, publish the container’s port 8501 to host's port 8501 where the TF serving listens to REST API requests
	* with `--name`, assign a name to the container for acessing later for checking status or killing it.
	* with `-v`,  mount the host local model directory `$TEST_DATA/saved_model_half_plus_two_mkl` on the container `/models/half_plus_two`.
	* with `-e`, setting an environment variable in the container which is read by TF serving
	* with `intel/intel-optimized-tensorflow-serving:latest` docker image
	```
	docker run \
	  -d \
	  -p 8501:8501 \
	  --name tfserving_half_plus_two \
	  -v $TEST_DATA/saved_model_half_plus_two_mkl:/models/half_plus_two \
	  -e MODEL_NAME=half_plus_two \
	  intel/intel-optimized-tensorflow-serving:latest
	```

* Query the model using the predict API:
	```
	curl -d '{"instances": [1.0, 2.0, 5.0]}' \
	-X POST http://localhost:8501/v1/models/half_plus_two:predict
	```
	You should see the following output:
	```
	{
	"predictions": [2.5, 3.0, 4.5]
	}
	```
	NOTE: If you see any issues as below after sending predict request, please make sure to set your proxy (inside corporate environment)
	```
	curl -d '{"instances": [1.0, 2.0, 5.0]}' \
		-X POST http://localhost:8501/v1/models/half_plus_two:predict \
		<http://localhost:8501/v1/models/half_plus_two:predict>
	<HTML>
	<HEAD><TITLE>Redirection</TITLE></HEAD>
	<BODY><H1>Redirect</H1></BODY>
	```
	Place this proxy information in your `~/.bashrc` or `/etc/environment`
	```
	export http_proxy="<http_proxy>"
	export https_proxy="<https_proxy>"
	export ftp_proxy="<ftp_proxy>"
	export socks_proxy="<socks_proxy>"
	export HTTP_PROXY=${http_proxy}
	export HTTPS_PROXY=${https_proxy}
	export FTP_PROXY=${ftp_proxy}
	export SOCKS_PROXY=${socks_proxy}
	export no_proxy=localhost,127.0.0.1,<add_your_machine_ip>,<add_your_machine_hostname>
	export NO_PROXY=${no_proxy}
	```

* After you are fininshed with querying, you can stop the container which is running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`.
	```
	docker rm -f tfserving_half_plus_two
	```

 *  **Note:** If you want to confirm that Intel® oneAPI Deep Neural Network Library (Intel® oneDNN) optimizations are being used, add `-e MKLDNN_VERBOSE=1` to the `docker run` command.   This will log Intel oneDNN messages in the docker logs, which you can inspect after a request is processed.
	```
	docker run \
	  -d \
	  -p 8501:8501 \
	  --name tfserving_half_plus_two \
	  -v $TEST_DATA/saved_model_half_plus_two_mkl:/models/half_plus_two \
	  -e MODEL_NAME=half_plus_two \
	  -e MKLDNN_VERBOSE=1 \
	  intel/intel-optimized-tensorflow-serving:latest
	```  
	 Query the model using the predict API as before:
    ```
    curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
    ```
    
    Result:
    ```
    {
        "predictions": [2.5, 3.0, 4.5]
    }
    ```
    
    Then, you should see the Intel oneDNN verbose output like below when you display the container's logs:
    ```
    docker logs tfserving_half_plus_two
    ```
    
    Output:
    ```
    ...
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nhwc out:f32_nChw16c,num:1,1x1x10x10,0.00488281     
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_hwio out:f32_OIhw16i16o,num:1,1x1x1x1,0.000976562
    mkldnn_verbose,exec,convolution,jit_1x1:avx512_common,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:x fdst:nChw16c,alg:convolution_direct,mb1_g1ic1oc1_ih10oh10kh1sh1dh0ph0_iw10ow10kw1sw1dw0pw0,0.00805664
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nChw16c out:f32_nhwc,num:1,1x1x10x10,0.012207
    ```

## Example: Serving ResNet-50 v1 Model

TensorFlow Serving requires the model to be in SavedModel format. In this example, we will :
* Download a pre-trained ResNet-50  v1 SavedModel 
* Use the [python client code](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example) from the TensorFlow Serving repository and query using two methods:
	* [Using REST API](#option-1-query-using-rest-api), which is simple to set up, but lacks performance when compared with gRPC
	* [Using gRPC](#option-2-query-using-grpc), which has optimal performance but the client code requires additional dependencies to be installed

**NOTE:** NCHW data format is optimal for Intel-optimized TensorFlow Serving.

#### Download and untar a ResNet-50 v1 SavedModel to `/tmp/resnet`
```
mkdir /tmp/resnet
curl -s http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NCHW_jpg.tar.gz \
| tar --strip-components=2 -C /tmp/resnet -xvz
```

### Option 1: Query using REST API 
* Querying using REST API is simple to set up, but lacks performance when compared with gRPC.

* If a running container is using port 8501, you need to stop it. View your running containers with `docker ps`.  To stop and remove the contatiner from the registry, copy the `CONTAINER ID` from the  `docker ps` output and run `docker rm -f <container_id>`.

* Start the container 
    * with `-d`, runs the container as a background process
	* with `-p`, publish the container’s port 8501 to host's port 8501 where the TF serving listens to REST API requests
	* with `--name`, assign a name to the container for acessing later for checking status or killing it.
	* with `-v`,  mount the host local model directory `/tmp/resnet` on the container `/models/resnet`.
	* with `-e`, setting an environment variable in the container which is read by TF serving
	* with `intel/intel-optimized-tensorflow-serving:latest` docker image
	```
	docker run \
	  -d \
	  -p 8501:8501 \
	  --name=tfserving_resnet_restapi \
	  -v "/tmp/resnet:/models/resnet" \
	  -e MODEL_NAME=resnet \
	  intel/intel-optimized-tensorflow-serving:latest
	```
* If you don't already have them, install the prerequisites for running the python client code
	```
	sudo apt-get install -y python python-requests
	```
* Run the example `resnet_client.py` script from the TensorFlow Serving repository
	```
	python $TF_SERVING_ROOT/tensorflow_serving/example/resnet_client.py
	```
	You should see the following output:
	```
	Prediction class: 286, avg latency: 34.7315 ms
	```
  **Note:** The real performance you see will depend on your hardware, environment, and whether or not you have configured the server parameters optimally. See the [General Best Practices](GeneralBestPractices.md) for more information.
  
* After you are fininshed with querying, you can stop the container which is running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`. 
	```
	docker rm -f tfserving_resnet_restapi
	```

### Option 2: Query using gRPC 
* Querying using gRPC will have optimal performance but the client code requires additional dependencies to be installed.

* If a running container is using port 8500, you need to stop it. View your running containers with `docker ps`.  To stop and remove the contatiner from the registry, copy the `CONTAINER ID` from the  `docker ps` output and run `docker rm -f <container_id>`.

* Start a container 
    * with `-d`, runs the container as a background process
	* with `-p`, publish the container’s port 8500 to host's port 8500 where the TF serving listens to gRPC requests
	* with `--name`, assign a name to the container for acessing later for checking status or killing it.
	* with `-v`,  mount the host local model directory `/tmp/resnet` on the container `/models/resnet`.
	* with `-e`, setting an environment variable in the container which is read by TF serving
	* with `intel/intel-optimized-tensorflow-serving:latest` docker image
 	```
    docker run \
	  -d \
	  -p 8500:8500 \
	  --name=tfserving_resnet_grpc \
	  -v "/tmp/resnet:/models/resnet" \
	  -e MODEL_NAME=resnet \
	  intel/intel-optimized-tensorflow-serving:latest
	```
* You will need a few python packages in order to run the client, we recommend installing them in a virtual environment. 
	```
	sudo apt-get install -y python python-pip
	pip install virtualenv
	```
* Create and activate the python virtual envirnoment. Install the packages needed for the gRPC client.
	```
	cd ~
	virtualenv -p python3 tfserving_venv
	source tfserving_venv/bin/activate
	pip install requests tensorflow tensorflow-serving-api
	```
* Run the example `resnet_client_grpc.py` script from the TensorFlow Serving repository, which you cloned earlier. 
  
  > Note: You may have to [migrate the script](https://www.tensorflow.org/guide/migrate) for TF2 compatibility, because it was not up to date last time we checked. 
    To fix the script, you can search-and-replace `tf.app` with `tf.compat.v1.app`.
	```
	python $TF_SERVING_ROOT/tensorflow_serving/example/resnet_client_grpc.py
	```
  You should see the similar output as below:
	```
	outputs {
	  key: "classes"
	  value {
	    dtype: DT_INT64
	    tensor_shape {
	      dim {
	        size: 1
	      }
	    }
	    int64_val: 286
	  }
	}
	outputs {
	  key: "probabilities"
	  value {
	    dtype: DT_FLOAT
	    tensor_shape {
	      dim {
	        size: 1
	      }
	      dim {
	        size: 1001
	      }
	    }
	    float_val: 7.8115895974e-08
	    float_val: 3.93756813821e-08
	    float_val: 6.0871172991e-07
	  .....
	  .....
	  }
	}
	model_spec {
	  name: "resnet"
	  version {
	    value: 1538686758
	  }
	  signature_name: "serving_default"
	}
	```
  
 
* To deactivate your virtual environment:
	```
	deactivate
	```

* After you are fininshed with querying, you can stop the container which is running in the background. To restart the container with the same name, you need to stop and remove the container from the registry. To view your running containers run `docker ps`. 
	```
	docker rm -f tfserving_resnet_grpc
	```


## Debugging

If you have any problems while making a request, the best way to debug is to check the docker logs.
First, find the Container ID of your running docker container with `docker ps` and then view its logs with `docker logs <container_id>`.
If you have added `-e MKLDNN_VERBOSE=1` to the `docker run` command, you should see mkldnn_verbose messages too.

