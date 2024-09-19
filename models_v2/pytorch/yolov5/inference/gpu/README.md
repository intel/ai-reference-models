# YOLO V5 Inference

[YOLOv5](https://github.com/ultralytics/yolov5) or the fifth iteration of You Only Look Once is a single-stage deep learning based object detection model. It is claimed to deliver real-time object detection with state-of-the-art accuracy. This directory contains a sample implementation of object detection with YOLOv5. It is targeted to run on Intel Discrete Graphics platforms (XPUs) by leveraging [Intel® Extension for Pytorch](https://github.com/intel/intel-extension-for-pytorch).

The sample supports two modes of execution:

* A performance benchmarking mode where the sample executes YOLOv5 inference based on dummy tensor initialization for a specified time duration, each of which in turn loops over a specified number of inputs or frames. The input is a tensor capturing a batch of input images that is fed to the YOLOv5 model for inference. The output is tensor representing the objects detected.
* A accuracy-check mode which takes in images from the [COCO 2017] validation dataset and returns the highest confidence score for the detected object.

The rest of this document covers more details about the model, dataset, and the control knobs for each mode of execution. Further, instructions are provided on how to use the scripts in this directory for execution in bare-metal and docker container environments.

## Model Information

[get_model.sh]: get_model.sh
[run_model.sh]: run_model.sh

This sample uses source code and weights from the reference Pytorch implementation by its authors taken at the following commit to drive the inference:

* <https://github.com/ultralytics/yolov5/commit/781401ec>

The weights and model need to be downloaded from the original implementation and installed under `yolov5` directory for the sample to be usable. The [get_model.sh] script may be used to accomplish this. The model run script ([run_model.sh]) is configured to invoke this script and automatically install `yolov5` directory if it is absent under working directory.

## Dataset

> [!NOTE]
> Throughtput and latency benchmarking recommended with dummy data (`./run_model.sh --dummy`). In such a case dataset setup can be skipped. Accuracy test with dataset (`./run_model.sh --data $DATASET_DIR`).

[COCO 2017]: https://cocodataset.org/#download

[COCO 2017] validation dataset is required to measure accuracy during inference. Visit [COCO 2017] site and download the following files:

* [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

[get_dataset.sh](get_dataset.sh) script can be used to download this file. Once downloaded, extract the archive and set `DATASET_DIR` environment variable to the dataset folder.

## Prerequisites

Hardware:

* **Flex Series** - [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
* **Max Series** - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)

Software:

* **Flex Series** - Intel® Data Center GPU Flex Series [Driver](https://dgpu-docs.intel.com/driver/installation.html)
* **Max Series** - Intel® Data Center GPU Max Series [Driver](https://dgpu-docs.intel.com/driver/installation.html)
* [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

## Run the model under container

Download the sample:

   ```
   git clone https://github.com/IntelAI/models.git
   cd models_v2/pytorch/yolov5/inference/gpu
   ```

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` under running container if you work behind the proxy.

Pull pre-built image with the sample:

```
docker pull intel/object-detection:pytorch-flex-gpu-yolov5-inference
```

or build it locally:

```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/flex-gpu/pytorch-yolov5-inference/pytorch-flex-series-yolov5-inference.Dockerfile \
  -t intel/object-detection:pytorch-flex-gpu-yolov5-inference .
```

Run sample as follows:

* With dummy data:

  * Running with dummy data is recommended for performance benchmarking (throughput and latency measurements)
  * Use higher `NUM_INPUTS` values for more precise peak performance results. `NUM_INPUTS` will be rounded to a multiple of `BATCH_SIZE`
  * **NOTE**: Accuracy will be zero when using dummy data

  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  export BATCH_SIZE=1
  export PLATFORM=Flex  # can also be: Max
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e MODEL_NAME=yolov5m \
    -e PLATFORM=${PLATFORM} \
    -e MAX_TEST_DURATION=60 \
    -e MIN_TEST_DURATION=60 \
    -e NUM_INPUTS=1 \
    -e BATCH_SIZE=${BATCH_SIZE} \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    intel/object-detection:pytorch-flex-gpu-yolov5-inference \
      /bin/bash -c "run_model.sh --dummy"
  ```

* With Coco dataset (assumes that dataset was downloaded to the `$DATASET_DIR` folder):

  * Running with dataset images is recommended for accuracy measurements but not recommended for benchmarking
  * **NOTE**: Performance results (throughput and latency measurements) may be impacted due to data handling overhead

  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  export BATCH_SIZE=1
  export PLATFORM=Flex  # can also be: Max
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e MODEL_NAME=yolov5m \
    -e PLATFORM=${PLATFORM} \
    -e NUM_INPUTS=5000 \
    -e BATCH_SIZE=${BATCH_SIZE} \
    -e OUTPUT_DIR=/tmp/output \
    -e DATASET_DIR=/dataset \
    -v /tmp/output:/tmp/output \
    -v $DATASET_DIR:/dataset \
    intel/object-detection:pytorch-flex-gpu-yolov5-inference \
      /bin/bash -c "run_model.sh --dummy"
  ```

Mind the following `docker run` arguments:

* HTTPS proxy is required to download model over network (`-e https_proxy=<...>`)
* `--cap-add SYS_NICE` is required for `numactl`
* `--device /dev/dri` is required to expose GPU device to running container
* `--ipc=host` is required for multi-stream benchmarking (`./run_model.sh --dummy --streams 2`) or large dataset cases
* `-v $DATASET_DIR:/dataset` in case where dataset is used. `$DATASET_DIR` should be replaced with the actual path (for example: `/home/<user>/val2017`) or the specific image (for example: `/home/<user>/val2017/000000025394.jpg`) to the Coco dataset.

## Run the model on baremetal

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` before running `run_model.sh` if you work behind proxy.

1. Download the sample:

   ```
   git clone https://github.com/IntelAI/models.git
   cd models_v2/pytorch/yolov5/inference/gpu
   ```

1. Create virtual environment `venv` and activate it:

   ```
   python3 -m venv venv
   . ./venv/bin/activate
   ```

1. Install sample python dependencies:

    ```
    ./setup.sh
    ```

1. Install [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)
1. Add path to common python modules in the repo:

   ```
   export PYTHONPATH=$(pwd)/../../../../common
   ```

1. Setup required environment variables and run the sample with `./run_model.sh`:

   * With dummy data:

     * Running with dummy data is recommended for performance benchmarking (throughput and latency measurements)
     * Use higher `NUM_INPUTS` values for more precise peak performance results. `NUM_INPUTS` will be rounded to a multiple of `BATCH_SIZE`
     * **NOTE**: Accuracy will be zero when using dummy data

     ```
     export MODEL_NAME=yolov5m
     export PLATFORM=Flex  # can also be: Max
     export BATCH_SIZE=1
     export NUM_INPUTS=1
     export OUTPUT_DIR=/tmp/output
     ./run_model.sh --dummy
     ```

   * With Coco dataset (assumes that dataset was downloaded to the `$DATASET_DIR` folder):

     * Running with dataset images is recommended for accuracy measurements
     * **NOTE**: Performance results (throughput and latency measurements) may be impacted due to data handling overhead

     ```
     export MODEL_NAME=yolov5m
     export PLATFORM=Flex  # can also be: Max
     export BATCH_SIZE=1
     export NUM_INPUTS=5000
     export OUTPUT_DIR=/tmp/output
     export DATASET_DIR=$DATASET_DIR
     ./run_model.sh
     ```

## Runtime arguments and environment variables

`run_model.sh` accepts a number of arguments to tune behavior. `run_model.sh` supports the use of environment variables as well as command line arguments for specifying these arguments (see the table below for details).

Before running `run_model.sh` script, user is required to:

* Set `OUTPUT_DIR` environment variable (or use `--output-dir`) where script should write logs.
* Use `--dummy` data or set `DATASET_DIR` environment variable (or use `--data`) pointing to Coco dataset.

Other arguments and/or environment variables are optional and should be used according to the actual needs (see examples above).

| Argument              | Environment variable | Valid Values      | Purpose                                                                 |
| ------------------    | -------------------- | ----------------- | ---------------------------------------------------------------------   |
| `--ipex`              | `IPEX`               | `yes`             | Use [Intel® Extension for Pytorch] for XPU support (default: `yes`)     |
|                       |                      | `no`              | Use PyTorch XPU backend instead of [Intel® Extension for Pytorch]. Requires PyTorch version 2.4.0a or later. |
| `--amp`               | `AMP`                | `no`              | Use AMP on model convertion to the desired precision (default: `no`)    |
|                       |                      | `yes`             |                                                                         |
| `--batch-size`        | `BATCH_SIZE`         | >=1               | Batch size to use (default: `1`)                                        |
| `--data`              | `DATASET_DIR`        | String            | Location to load images from                                            |
| `--dummy`             | `DUMMY`              |                   | Use randomly generated dummy dataset in place of `--data` argument      |
| `--load`              | `LOAD_PATH`          |                   | Local path to load model from (default: disabled)                       |
| `--num-inputs`        | `NUM_INPUTS`         | >=1               | Number of images to load (default: `1`)                                 |
| `--max-test-duration` | `MAX_TEST_DURATION`  | >=0               | Maximum duration in seconds to run benchmark. Testing will be truncated |
|                       |                      |                   | once maximum test duration has been reached. (default: disabled)        |
| `--min-test-duration` | `MIN_TEST_DURATION`  | >=0               | Minimum duration in seconds to run benchmark. Images will be repeated   |
| `--output-dir`        | `OUTPUT_DIR`         | String            | Location to write output                                                |
| `--proxy`             | `https_proxy`        | String            | System proxy                                                            |
| `--precision`         | `PRECISION`          | `fp16`            | Precision to use for the model (default: `fp32`)                        |
|                       |                      | `fp32`            |                                                                         |
| `--save`              | `SAVE_PATH`          |                   | Local path to save model to (default: disabled)                         |
| `--streams`           | `STREAMS`            | >=1               | Number of parallel streams to do inference on (default: `1`)            |
| `--platform`          | `PLATFORM`           | `Flex`            | Platform that inference is being ran on                                 |
|                       |                      | `Max`             |                                                                         |
| `--socket`            | `SOCKET`             | String            | Socket to control telemetry capture (default: disabled)                 |

For more details, check help with `run_model.sh --help`

## Example output

Script output is written to the console as well as to the output directory in the file `output.log`.

For multi-stream cases per-stream results can be found in the `results_[STREAM_INSTANCE].json` files.

Final results of the inference run can be found in `results.yaml` file. More verbose results summaries are in `results.json` file.

The yaml file contents will look like:

```
results:
 - key: throughput
   value: 173.37709471804496
   unit: img/s
 - key: latency
   value: 5.76777458190918
   unit: ms
 - key: accuracy
   value: 0.83
   unit: confidence
```

## Performance Benchmarking

[benchmark.sh] script can be used to benchmark YOLOv5 performance for the [predefined use cases](profiles/README.md). The [benchmark.sh] script is a tiny YOLOv5 specific wrapper on top of [benchmark.py](/models_v2/common/benchmark.py) script. The workflow for running a benchmark is as follows:

* (optional) Specify path to [svr-info](https://github.com/intel/svr-info):

  ```
  export PATH_TO_SVR_INFO=/path/to/svrinfo
  ```

* Specify path to output benchmark results (folder must be creatable/writable under `root`):

  ```
  export OUTPUT_DIR=/opt/output
  ```

* Run the benchmark script (assumes ``intel/object-detection:pytorch-flex-gpu-yolov5-inference`` has already been pulled or built locally):

  ```
  sudo \
    PATH=$PATH_TO_SVR_INFO:$PATH \
    IMAGE=intel/object-detection:pytorch-flex-gpu-yolov5-inference \
    OUTPUT_DIR=$OUTPUT_DIR \
    PROFILE=$(pwd)/models_v2/pytorch/yolov5/inference/gpu/profiles/yolov5m.fp16.csv \
    PYTHONPATH=$(pwd)/models_v2/common \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^//') \
    $(pwd)/models_v2/pytorch/yolov5/inference/gpu/benchmark.sh
  ```

* Final output will be written to ``$OUTPUT_DIR``.

> [!NOTE]
> Additonal arguments that arent specified in the benchmark profile (``yolov5m.fp16.csv`` in the example above) can be specified through environment variables as described in previous sections.

## Usage With CUDA GPU

Scripts have a matching degree of functionality for usage on CUDA GPU's. However, this is significantly less validated and so may not work as smoothly. The primary difference for using these scripts with CUDA is building the associated docker image. We will not cover CUDA on baremetal here. In addition Intel does not provide pre-built dockers for CUDA. These must be built locally.

```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/cuda-gpu/pytorch-yolov5-inference/pytorch-cuda-series-yolov5-inference.Dockerfile \
  -t intel/object-detection:pytorch-cuda-gpu-yolov5-inference .
```

All other usage outlined in this README should be identical, with the exception of referencing this CUDA docker image in place of the Intel GPU when running `docker run` as well as needing to add the `--gpus all` argument.

Example usage with dummy data is shown below:

```
mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
export BATCH_SIZE=1
docker run -it --rm --gpus all --ipc=host \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
  --cap-add SYS_NICE \
  --device /dev/dri/ \
  -e MODEL_NAME=yolov5m \
  -e PLATFORM=CUDA \
  -e MAX_TEST_DURATION=60 \
  -e MIN_TEST_DURATION=60 \
  -e NUM_INPUTS=1 \
  -e BATCH_SIZE=${BATCH_SIZE} \
  -e OUTPUT_DIR=/tmp/output \
  -v /tmp/output:/tmp/output \
  intel/object-detection:pytorch-cuda-gpu-yolov5-inference \
  /bin/bash -c "./run_model.sh --dummy"
```
