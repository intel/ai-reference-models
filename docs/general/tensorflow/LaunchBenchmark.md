# Launch Benchmark Script

The TensorFlow inference tutorials use the Model Zoo's [`launch_benchmark.py`](/benchmarks/launch_benchmark.py) script, 
and while this is convenient for illustrating the most important concepts in the tutorials and ensuring that models will run on any platform, 
it obscures many details about the environment, dependencies, and data pre- and post-processing for each model.
In a future release, this will be remedied with simplified tutorial routines that are more self-explanatory, but until then,
please refer to the following description of how the launch script and other scripts work to set up and run models.
Below the general description is an [index of links](#model-scripts-for-tensorflow-fp32-inference) that point to the relevant files for each model, should you need them. 

## How it Works

1. The script [`launch_benchmark.py`](/benchmarks/launch_benchmark.py) pulls a docker image specified by the script's `--docker-image` argument and runs a container. 
   [Here](#launch_benchmarkpy-flags) is the full list of available flags. To run benchmarking without a docker container,
   see the [bare metal instructions](#alpha-feature-running-on-bare-metal).
2. The container's entrypoint script [`start.sh`](/benchmarks/common/tensorflow/start.sh) installs required dependencies, e.g. python packages and `numactl`, and sets the PYTHONPATH environment variable to point to the required dependencies. 
   [Here](#startsh-flags) is the full list of available flags.
3. The [`run_tf_benchmark.py`](/benchmarks/common/tensorflow/run_tf_benchmark.py) script calls the model's initialization routine, called `model_init.py` (see [here](#model-scripts-for-tensorflow-fp32-inference)).
   The `model_init.py` is different for every model and sets environment variables (like `KMP_BLOCKTIME`, `KMP_SETTINGS`, `KMP_AFFINITY`, and `OMP_NUM_THREADS`) to the best known settings. It also sets `num_inter_threads` and `num_intra_threads` to the best known settings, if the user has not set them explicitly.
4. Then `model_init.py` creates the command to call the model's inference script, including the `numactl` prefix. This inference script invokes a TensorFlow session and is also different for every model (see [here](#model-scripts-for-tensorflow-fp32-inference)).
5. The inference script calls preprocessing modules (usually but not always named `preprocessing.py`) to prepare the data with any pre- or post-processing routines required by the model.

## Model Scripts for TensorFlow FP32 Inference

* Image Recognition
    * ResNet50: [init](/benchmarks/image_recognition/tensorflow/resnet50/inference/fp32/model_init.py) | 
                [inference](/models/image_recognition/tensorflow/resnet50/fp32/eval_image_classifier_inference.py) | 
                [preprocessing](/models/image_recognition/tensorflow/resnet50/fp32/preprocessing.py) 
    * ResNet101: [init](/benchmarks/image_recognition/tensorflow/resnet101/inference/fp32/model_init.py) | 
                 [inference](/models/image_recognition/tensorflow/resnet101/fp32/benchmark.py) | 
                 [preprocessing](/models/image_recognition/tensorflow/resnet101/fp32/preprocessing.py) 
    * InceptionV3: [init](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/model_init.py) | 
                   [inference](/models/image_recognition/tensorflow/inceptionv3/fp32/eval_image_classifier_inference.py) | 
                   [preprocessing](/models/image_recognition/tensorflow/inceptionv3/fp32/preprocessing.py) 
* Image Segmentation
    * 3D U-Net: [init](/benchmarks/image_segmentation/tensorflow/3d_unet/inference/fp32/model_init.py) | 
                [inference](/models/image_segmentation/tensorflow/3d_unet/inference/fp32/brats/predict.py) | 
                [preprocessing](/models/image_segmentation/tensorflow/3d_unet/inference/fp32/unet3d) 
* Recommendation Systems
    * Wide and Deep: [init](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/fp32/model_init.py) | 
                [inference](/models/recommendation/tensorflow/wide_deep_large_ds/inference/inference.py) | 
                [preprocessing](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py) 
     
## ```launch_benchmark.py``` flags   

```
positional arguments:
  model_args            Additional command line arguments (prefix flag start
                        with '--').

optional arguments:
  -h, --help            show this help message and exit
  -f FRAMEWORK, --framework FRAMEWORK
                        Specify the name of the deep learning framework to
                        use.
  -r [MODEL_SOURCE_DIR], --model-source-dir [MODEL_SOURCE_DIR]
                        Specify the models source directory from your local
                        machine
  -p {fp32,int8,bfloat16}, --precision {fp32,int8,bfloat16}
                        Specify the model precision to use: fp32, int8, or
                        bfloat16
  -mo {training,inference}, --mode {training,inference}
                        Specify the type training or inference
  -m MODEL_NAME, --model-name MODEL_NAME
                        model name to run benchmarks for
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Specify the batch size. If this parameter is not
                        specified or is -1, the largest ideal batch size for
                        the model will be used
  -d DATA_LOCATION, --data-location DATA_LOCATION
                        Specify the location of the data. If this parameter is
                        not specified, the benchmark will use random/dummy
                        data.
  -i SOCKET_ID, --socket-id SOCKET_ID
                        Specify which socket to use. Only one socket will be
                        used when this value is set. If used in conjunction
                        with --num-cores, all cores will be allocated on the
                        single socket.
  -n NUM_CORES, --num-cores NUM_CORES
                        Specify the number of cores to use. If the parameter
                        is not specified or is -1, all cores will be used.
  -a NUM_INTRA_THREADS, --num-intra-threads NUM_INTRA_THREADS
                        Specify the number of threads within the layer
  -e NUM_INTER_THREADS, --num-inter-threads NUM_INTER_THREADS
                        Specify the number threads between layers
  --data-num-intra-threads DATA_NUM_INTRA_THREADS
                        The number intra op threads for the data layer config
  --data-num-inter-threads DATA_NUM_INTER_THREADS
                        The number inter op threads for the data layer config
  -v, --verbose         Print verbose information.
  --docker-image DOCKER_IMAGE
                        Specify the docker image/tag to use
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Specify the location of trained model checkpoint
                        directory. If mode=training model/weights will be
                        written to this location. If mode=inference assumes
                        that the location points to a model that has already
                        been trained.
  -k, --benchmark-only  For benchmark measurement only. If neither
                        --benchmark-only or --accuracy-only are specified, it
                        will default to run benchmarking.
  --accuracy-only       For accuracy measurement only. If neither --benchmark-
                        only or --accuracy-only are specified, it will default
                        to run benchmarking.
  --output-results      Writes inference output to a file, when used in
                        conjunction with --accuracy-only and --mode=inference.
  --output-dir OUTPUT_DIR
                        Folder to dump output into.
  -g INPUT_GRAPH, --in-graph INPUT_GRAPH
                        Full path to the input graph
  --debug               Launches debug mode which doesn't execute start.sh
```

## Alpha feature: Running on bare metal

We recommend using [Docker](https://www.docker.com) to run the
benchmarking scripts, as that provides a consistent environment where
the script can install all the necessary dependencies to run the models
in this repo. For this reason, the tutorials and model README files
provide instructions on how to run the model in a Docker container.
However, if you need to run without Docker, the instructions below
describe how that can be done using the `launch_benchmark.py` script.

### Prerequisites for running on bare metal

Since the `launch_benchmark.py` is intended to run in an Ubuntu-based
Docker container, running on bare metal also will only work when running
on Ubuntu.

Before running benchmarking, you must also install all the dependencies
that are required to run the model.

Basic requirements for running all models include:
 * python (If the model's README file specifies to use a python3 TensorFlow docker image, then use python 3 on bare metal, otherwise use python 2.7)
 * [intel-tensorflow](https://github.com/tensorflow/tensorflow/blob/master/README.md#community-supported-builds)
 * python-tk
 * numactl
 * libsm6
 * libxext6
 * requests

Individual models may have additional dependencies that need to be
installed. The easiest way is to find this out find the model's function in
the [start.sh](/benchmarks/common/tensorflow/start.sh) script and check
if any additional dependencies are being installed. For example, many of
the Object Detection models require the python
[cocoapi](https://github.com/cocodataset/cocoapi) and dependencies
from a [requirements.txt file](/benchmarks/object_detection/tensorflow/ssd-mobilenet/requirements.txt)
to be installed.

### Running the launch script on bare metal

Once you have installed all of the requirements necessary to run the
model, you can follow the [tutorials](/docs/README.md) or model
[README](/benchmarks/README.md) files for instructions on getting the
required code repositories, dataset, and pretrained model. Once you get
to the step for running the `launch_benchmark.py` script, omit the
`--docker-image` arg to run without a Docker container. If you have
installed the model dependencies in a virtual environment be sure that
you are calling the proper python executable, which includes the
dependencies that you installed in the previous step.

Also, note that if you are using the same clone of the Model Zoo, which
you previously used with docker, you may need to change the owner
on your log directory, or run with `sudo` in order for the `tee`
commands writing to the log file to work properly.

For example, in order to run ResNet50 FP32 benchmarking on bare metal,
the following command can be used:

```
 /home/<user>/venv/bin/python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --socket-id 0
```
