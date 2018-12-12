# Intel-Optimized TensorFlow Serving Installation (Linux)

## Pre-installation

Whether you are installing TensorFlow Serving on baremetal or a cloud VM, the machine/instance should have at least 20 GB of free disk space (required) and at least 8 logical cores (highly recommended).
It also needs to have HTTP/S traffic enabled and you must be able to login using SSH (for details, see your system administrator or cloud provider console documentation).
We recommend Ubuntu 16.04 as these instructions were written and tested for it, but the process should be very similar for any other Linux distribution.

For a Google Cloud compute instance, you may find these pages helpful:
* [https://cloud.google.com/compute/docs/instances/create-start-instance](https://cloud.google.com/compute/docs/instances/create-start-instance)
* [https://cloud.google.com/sdk/gcloud/reference/compute/ssh](https://cloud.google.com/sdk/gcloud/reference/compute/ssh)

For a AWS compute instance, you may find these pages helpful:
* [https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
* [https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html)

## Installation

1. SSH into the machine or VM.
2. Install Docker and run the hello-world test to verify it is working properly. Instructions for Ubuntu can be found at
[https://docs.docker.com/install/linux/docker-ce/ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu) and you can use the menu on the left of that page if you are using a different Linux distribution.
3. Add your username to the docker group:
   ```
   $ sudo usermod -aG docker `whoami`
   ```
4. Exit and restart your SSH session so that your username is in effect in the docker group.
5. Download *Dockerfile.mkl* and *Dockerfile.devel-mkl* from [here](https://github.com/Intel-tensorflow/serving/tree/intel/mkl_dockerfiles/tensorflow_serving/tools/docker).

Next, we will do two docker builds. **NOTE 1**: they both require a `.` path argument at the end; see
[https://docs.docker.com/engine/reference/commandline/build/#examples](https://docs.docker.com/engine/reference/commandline/build/#examples) for more background.
**NOTE 2**: if your machine is behind a proxy, you will need to pass proxy arguments to both build commands. For example:
```
--build-arg http_proxy="your company http proxy" --build-arg https_proxy="your company https proxy"
```

6. Build *Dockerfile.devel-mkl* which contains all the required development tools to build sources. This build will take the longest.
   ```
   $ docker build -f Dockerfile.devel-mkl -t tensorflow/serving:latest-devel-mkl .
   ```
7. Finally, build *Dockerfile.mkl* which creates a light-weight image without any development tools in it.
   ```
   $ docker build -f Dockerfile.mkl -t tensorflow/serving:mkl .
   ```

## Testing the Server

Run TensorFlow Serving using the above image and test it using the official [TensorFlow Serving example](https://www.tensorflow.org/serving/).
- Clone the repository:
  ```
  $ git clone https://github.com/tensorflow/serving
  ```
- Set the location of test models:
  ```
  $ TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
  ```
- Start the TensorFlow Serving container and open the REST API port:
  ```
  $ docker run -t --rm -p 8501:8501 -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving:mkl &
  ```
  Note: If you want to confirm that MKL optimizations are being used, add `-e MKLDNN_VERBOSE=1` to the `docker run` command.
  This will log MKL messages in the docker logs, which you can inspect after a request is processed.

- Query the model using the predict API:
  ```
  $ curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
  ```
- Output:
  ```
  {
    "predictions": [2.5, 3.0, 4.5]
  }
  ```
- Finally, since the container is running in the background, you will need to stop it. View your running containers with `docker ps`.
To stop one, copy the Container ID and run `docker stop <container_id>`.

## ResNet50 Examples

The following examples use [pre-trained models](https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model)
and [client code](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example) from the TensorFlow repository.
NOTE: NCHW data format is optimal for Intel-optimized TensorFlow Serving.

* Download a ResNet50 saved model:
  ```
  $ wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NCHW_jpg.tar.gz
  ```
* Untar it:
  ```
  $ tar -xvzf resnet_v1_fp32_savedmodel_NCHW_jpg.tar.gz
  ```

### Option 1: Using the REST API (simple to set up, but performance is not as good as GRPC)

* If a running container is using port 8501, you need to stop it. View your running containers with `docker ps`.
To stop one, copy the Container ID and run `docker stop <container_id>`.
* Run the image with the saved model using port 8501:
  ```
  $ docker run --name=tfserving_mkl --rm -d -p 8501:8501 -v "$(pwd)/resnet_v1_fp32_savedmodel_NCHW_jpg:/models/resnet" -e MODEL_NAME=resnet tensorflow/serving:mkl &
  ```
* Run the example resnet_client.py script from the TensorFlow Serving repository,
which you should already have cloned in your home directory (if not, see the half_plus_two example):
  ```
  python serving/tensorflow_serving/example/resnet_client.py
  ```
* Output:
  ```
  Prediction class: 286, avg latency: 34.7315 ms
  ```
  Note: The real avg latency you see will depend on your hardware, environment, and whether or not you have configured the server parameters optimally. See the [General Best Practices](GeneralBestPractices.md) for more information.

### Option 2: Using GRPC (this is the fastest method, but the client has more dependencies)

* If a running container is using port 8500, you need to stop it. View your running containers with `docker ps`.
To stop one, copy the Container ID and run `docker stop <container_id>`.
* Run the image with the saved model using port 8500:
  ```
  $ docker run --name=tfserving_mkl --rm -d -p 8500:8500 -v "$(pwd)/resnet_v1_fp32_savedmodel_NCHW_jpg:/models/resnet" -e MODEL_NAME=resnet tensorflow/serving:mkl &
  ```
* You will need a few python packages in order to run the client, so if they are not already on your VM, we recommend installing them in a virtual environment.
Get pip if you do not already have it, install virtualenv if not available, create an environment, and activate it.
  ```
  $ sudo apt-get install python-pip
  $ pip install virtualenv
  $ virtualenv venv
  $ source venv/bin/activate
  ```
* Install the packages needed for the GRPC client.
  ```
  (venv)$ pip install grpc
  (venv)$ pip install requests
  (venv)$ pip install tensorflow
  (venv)$ pip install tensorflow-serving-api
  ```
  Note: Although not necessary for running the client code, you can use the latest
  [Intel Optimization for TensorFlow pip wheel file](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#pip_27)
  instead of installing the default tensorflow.

* Run the example resnet_client_grpc.py script from the TensorFlow Serving repository, which you should already have cloned in your home directory (if not, see the half_plus_two example).
  ```
  (venv)$ python serving/tensorflow_serving/example/resnet_client_grpc.py
  ```
  Output:
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
      float_val: ...
      ...
    }
  }
  model_spec {
    name: "resnet"
    version {
      value: 1538686847
    }
    signature_name: "serving_default"
  }
  ```
* To deactivate your virtual environment:
  ```
  (venv)$ deactivate
  ```

## Debugging

If you have any problems while making a request, the best way to debug is to check the docker logs.
First, find the Container ID of your running docker container with `docker ps` and then view its logs with `docker logs <container_id>`.
If you have added `-e MKLDNN_VERBOSE=1` to the `docker run` command, you should see mkldnn_verbose messages too.
