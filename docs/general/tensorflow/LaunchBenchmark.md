# Launch Benchmark Script

The TensorFlow inference tutorials use the Model Zoo's [`launch_benchmark.py`](/benchmarks/launch_benchmark.py) script, 
and while this is convenient for illustrating the most important concepts in the tutorials and ensuring that models will run on any platform, 
it obscures many details about the environment, dependencies, and data pre- and post-processing for each model.
In a future release, this will be remedied with simplified tutorial routines that are more self-explanatory, but until then,
please refer to the following description of how the launch script and other scripts work to set up and run models.
Below the general description is an [index of links](#model-scripts-for-tensorflow-fp32-inference) that point to the relevant files for each model, should you need them. 

## How it Works

1. The script [`launch_benchmark.py`](/benchmarks/launch_benchmark.py) pulls a docker image specified by the script's `--docker-image` argument and runs a container. 
   [Here](#launch_benchmarkpy-flags) is the full list of available flags. To run a model without a docker container,
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
                [inference](/models/image_recognition/tensorflow/resnet50/inference/eval_image_classifier_inference.py) | 
                [preprocessing](/models/image_recognition/tensorflow/resnet50/inference/preprocessing.py) 
    * ResNet101: [init](/benchmarks/image_recognition/tensorflow/resnet101/inference/fp32/model_init.py) | 
                 [inference](/models/image_recognition/tensorflow/resnet101/inference/eval_image_classifier_inference.py) | 
                 [preprocessing](/models/image_recognition/tensorflow/resnet101/inference/preprocessing.py) 
    * InceptionV3: [init](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/model_init.py) | 
                   [inference](/models/image_recognition/tensorflow/inceptionv3/fp32/eval_image_classifier_inference.py) | 
                   [preprocessing](/models/image_recognition/tensorflow/inceptionv3/fp32/preprocessing.py) 
* Language Translation
    * Transformer-LT: [init](/benchmarks/language_translation/tensorflow/transformer_lt_official/inference/fp32/model_init.py) | 
                [inference](/models/language_translation/tensorflow/transformer_lt_official/inference/fp32/infer_ab.py)    
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
  -p {fp32,int8}, --precision {fp32,int8}
                        Specify the model precision to use: fp32, int8
  -mo {training,inference}, --mode {training,inference}
                        Specify the type training or inference
  -m MODEL_NAME, --model-name MODEL_NAME
                        model name to run
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Specify the batch size. If this parameter is not
                        specified or is -1, the largest ideal batch size for
                        the model will be used
  -ts NUM_TRAIN_STEPS, --num-train-steps NUM_TRAIN_STEPS
                        Specify the number of training steps
  --mpi_num_processes MPI
                        The number of MPI processes. This cannot in
                        conjunction with --numa-cores-per-instance, which uses
                        numactl to run multiple instances.
  --mpi_num_processes_per_socket NUM_MPI
                        Specify how many MPI processes to launch per socket
  --numa-cores-per-instance NUMA_CORES_PER_INSTANCE
                        If set, the script will run multiple instances using
                        numactl to specify which cores will be used to execute
                        each instance. Set the value of this arg to a positive
                        integer for the number of cores to use per instance or
                        to 'socket' to indicate that all the cores on a socket
                        should be used for each instance. This cannot be used
                        in conjunction with --mpi_num_processes, which uses
                        mpirun.
  -d DATA_LOCATION, --data-location DATA_LOCATION
                        Specify the location of the data. If this parameter is
                        not specified, the script will use random/dummy
                        data.
  -i SOCKET_ID, --socket-id SOCKET_ID
                        Specify which socket to use. Only one socket will be
                        used when this value is set. If used in conjunction
                        with --num-cores, all cores will be allocated on the
                        single socket.
  --num-instances NUM_INSTANCES
                        Specify the number of instances to run. This flag is
                        deprecated and will be removed in the future. Please
                        use --numa-cores-per-instance instead.
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
  -bb BACKBONE_MODEL, --backbone_model BACKBONE_MODEL
                        Specify the location of backbone-model directory.
                        This option can be used by models (like SSD_Resnet34)
                        to do fine-tuning training or achieve convergence.
  -k, --benchmark-only  For performance measurement only. If neither
                        --benchmark-only or --accuracy-only are specified, it
                        will default to run for performance.
  --accuracy-only       For accuracy measurement only. If neither --benchmark-
                        only or --accuracy-only are specified, it will default
                        to run for performance.
  --output-results      Writes inference output to a file, when used in
                        conjunction with --accuracy-only and --mode=inference.
  --output-dir OUTPUT_DIR
                        Folder to dump output into.
  --tf-serving-version TF_SERVING_VERSION
                        Specify the version of tensorflow serving.
                        If nothing is specified, it defaults to master
                        of tensorflow serving.
  --disable-tcmalloc {True,False}
                        When TCMalloc is enabled, the google-perftools are
                        installed (if running using docker) and the LD_PRELOAD
                        environment variable is set to point to the TCMalloc
                        library file. The TCMalloc memory allocator produces
                        better performance results with smaller batch sizes.
                        This flag disables the use of TCMalloc when set to
                        True. For int8 benchmarking, TCMalloc is enabled by
                        default (--disable-tcmalloc=False). For other
                        precisions, the flag is --disable-tcmalloc=True by
                        default.
  --tcmalloc-large-alloc-report-threshold TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
                        Sets the TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
                        environment variable to the specified value. The
                        environment variable sets the threshold (in bytes) for
                        when large memory allocation messages will be
                        displayed.
  -g INPUT_GRAPH, --in-graph INPUT_GRAPH
                        Full path to the input graph
  --volume CUSTOM_VOLUMES
                        Specify a custom volume to mount in the container,
                        which follows the same format as the docker --volume
                        flag (https://docs.docker.com/storage/volumes/). This
                        argument can only be used in conjunction with a
                        --docker-image.
  --debug               Launches debug mode which doesn't execute start.sh
  --noinstall           Whether to install packages for a given model when
                        running in docker (default --noinstall='False') or on
                        bare metal (default --noinstall='True')
  --dry-run             Shows the call to the model without actually running
                        it
  --weight-sharing      Supports experimental weight-sharing feature for RN50
                        int8/bf16 inference only
                      
```

## Volume mounts

When running the launch script using a docker image, volumes will
automatically get mounted in the container for the following
directories:

| Directory | Mount location in the container |
|-----------|---------------------------------|
| Model zoo `/benchmarks` code | `/workspace/benchmarks` |
| Model zoo `/models` code | `/workspace/intelai_models` |
| `--model-source-dir` code | `/workspace/models` |
| `--checkpoints` directory | `/checkpoints` |
| `--in-graph` file | `/in_graph` |
| `--dataset-location` | `/dataset` |

If you would like additional directories mounted in the docker
container, you can specify them by using the `--volume` flag using the
same `:` separated field format [as docker](https://docs.docker.com/storage/volumes/).
For example, the following command will mount `/home/<user>/custom_folder_1`
in the container at `custom_folder_1` and `/home/<user>/custom_folder_2`
in the container at `custom_folder_2`:

```
python launch_benchmark.py \
        --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
        --model-name resnet50 \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --batch-size 1 \
        --socket-id 0 \
        --data-location /home/<user>/Imagenet_Validation \
        --docker-image intel/intel-optimized-tensorflow:latest \
        --volume /home/<user>/custom_folder_1:/custom_folder_1 \
        --volume /home/<user>/custom_folder_2:/custom_folder_2
```

Note that volume mounting only applies when running in a docker
container. When running on [bare metal](#alpha-feature-running-on-bare-metal),
files are accessed in their original location.

## Debugging

The `--debug` flag in the `launch_benchmarks.py` script gives you a
shell into the docker container with the [volumes mounted](#volume-mounts)
for any dataset, pretrained model, model source code, etc that has been
provided by the other flags. It does not execute the `start.sh` script,
and is intended as a way to setup an environment for quicker iteration
when debugging and doing development. From the shell, you can manually
execute the `start.sh` script and select to not re-install dependencies
each time that you re-run, so that the script takes less time to run.

Below is an example showing how to use the `--debug` flag:

1. Run the model using your model's `launch_benchmark.py` command, but
   add on the `--debug` flag, which will take you to a shell. If you
   list the files in the directory at that prompt, you will see the
   `start.sh` file:

   ```
   python launch_benchmark.py \
        --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
        --model-name resnet50 \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --batch-size=1 \
        --socket-id 0 \
        --data-location /home/<user>/Imagenet_Validation \
        --docker-image intel/intel-optimized-tensorflow:latest \
        --debug

   # ls
   __init__.py  logs  run_tf_benchmark.py  start.sh
   ```

2. Flags that were passed to the launch script are set as environment
   variables in the container:

   ```
   # env
   EXTERNAL_MODELS_SOURCE_DIRECTORY=None
   IN_GRAPH=/in_graph/resnet50_fp32_pretrained_model.pb
   WORKSPACE=/workspace/benchmarks/common/tensorflow
   MODEL_NAME=resnet50
   PRECISION=fp32
   BATCH_SIZE=1
   MOUNT_EXTERNAL_MODELS_SOURCE=/workspace/models
   DATASET_LOCATION=/dataset
   BENCHMARK_ONLY=True
   ACCURACY_ONLY=False
   ...
   ```
3. Run the `start.sh` script, which will setup the `PYTHONPATH`, install
   dependencies, and then run the model:
   ```
   # bash start.sh
   ...
   Iteration 48: 0.011513 sec
   Iteration 49: 0.011664 sec
   Iteration 50: 0.011802 sec
   Average time: 0.011650 sec
   Batch size = 1
   Latency: 11.650 ms
   Throughput: 85.833 images/sec
   Ran inference with batch size 1
   Log location outside container: <output directory>/benchmark_resnet50_inference_fp32_20190403_212048.log
   ```

4. Code changes that are made locally will also be made in the container
   (and vice versa), since the directories are mounted in the docker
   container. Once code changes are made, you can rerun the start
   script, except set the `NOINSTALL` variable, since dependencies were
   already installed in the previous run. You can also change the
   environment variable values for other settings, like the batch size.

   ```
   # NOINSTALL=True
   # BATCH_SIZE=128
   # bash start.sh
   ...
   Iteration 48: 0.631819 sec
   Iteration 49: 0.625606 sec
   Iteration 50: 0.618813 sec
   Average time: 0.625285 sec
   Batch size = 128
   Throughput: 204.707 images/sec
   Ran inference with batch size 128
   Log location outside container: <output directory>/benchmark_resnet50_inference_fp32_20190403_212310.log
   ```

5. Once you are done with the session, exit out of the docker container:
   ```
   # exit
   ```

## Alpha feature: Running on bare metal

We recommend using [Docker](https://www.docker.com) to run the
model scripts, as that provides a consistent environment where
the script can install all the necessary dependencies to run the models
in this repo. For this reason, the tutorials and model README files
provide instructions on how to run the model in a Docker container.
However, if you need to run without Docker, the instructions below
describe how that can be done using the `launch_benchmark.py` script.

### Prerequisites for running on bare metal

Before running a model, you must also install all the dependencies
that are required to run that model. **(Note: the `--noinstall` 
flag defaults to 'True' when running on bare metal.)**

Basic requirements for running all models include:
 * python 3.6
 * [intel-tensorflow](https://github.com/tensorflow/tensorflow/blob/master/README.md#community-supported-builds)
 * python-tk
 * libsm6
 * libxext6
 * requests
 * numactl
 
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
`--docker-image` arg to run without a Docker container.
If you run on Windows, please check the [instructions for the environment setup and running the available models](Windows.md).
If you have installed the model dependencies in a virtual environment be sure that
you are calling the proper python executable, which includes the
dependencies that you installed in the previous step.

Also, note that if you are using the same clone of the Model Zoo, which
you previously used with docker, you may need to change the owner
on your log directory, or run with `sudo` in order for the `tee`
commands writing to the log file to work properly.

For example, in order to run ResNet50 FP32 on bare metal,
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

> When running on bare metal, be aware of environment variables that you
have set on your system. The model zoo scripts intentionally do not
overwrite environment variables that have already been set, such as
`OMP_NUM_THREADS`. The same is true when running in a docker container,
but since a new docker container instance is started with each run, you
won't have previously set environment variables, like you may have on
bare metal.
