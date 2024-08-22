[FBNet]: https://arxiv.org/abs/1812.03443
[Intel® Extension for Pytorch]: https://github.com/intel/intel-extension-for-pytorch
[FBNet Model C100]: https://github.com/huggingface/pytorch-image-models/blob/67b0b3d7c7da3dbd76f30375b086ba4a0656811f/timm/models/efficientnet.py#L1538
[HuggingFace README]: https://github.com/huggingface/pytorch-image-models/blob/main/README.md
[fbnetc_100.rmsp_in1k]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth
[Intel® Data Center GPU Flex Series]: https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html
[Driver]: https://dgpu-docs.intel.com/driver/installation.html
[ImageNet]: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
[torchvision.datasets.ImageNet]: https://pytorch.org/vision/0.16/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet
[ILSVRC2012_img_val.tar]: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
[ILSVRC2012_devkit_t12.tar.gz]: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
[get_dataset.sh]: get_dataset.sh
[benchmark.sh]: benchmark.sh

# FBNet Model Inference

[FBNet] is a convolutional neural network architecture that utilises depthwise convolutions and an inverted residual structure for image classification. This directory contains a sample implementation of image classification with FBNet. It is targeted to run on Intel Discrete Graphics platforms (XPUs) by leveraging [Intel® Extension for Pytorch].

The input to the model is a tensor representing a batch of input images that is fed to the FBNet model for inference. The output is a tensor representing the inputs classification results for all possible classes.

The sample supports two modes of execution:

* A performance benchmarking mode where the sample executes FBNet inference based on a dummy tensor of the requested batch size over a specified number of input frames. This dummy tensor dataset is itself looped over repeatedly until a minimum test duration has been reached. Average throughput and latency values are reported.
* A accuracy-check mode which takes in frames from the ImageNet 2012 validation dataset and measures the accuracy of the resulting classification against references contained in the dataset. A percentage score - both top 1 accuracy and top 5 accuracy values - of passing frames along with average throughput and latency values are reported.

The rest of this document covers more details about the model, dataset, and the control knobs for each mode of execution. Further, instructions are provided on how to use the scripts in this directory for execution in bare-metal and docker container environments.

# Model and Sources

The sample uses FBNet [model implementations from HuggingFace][FBNet Model C100]:

| Model           | Documentation                        | Weights                                 |
| --------------- | ------------------------------------ | --------------------------------------- |
| fbnetc_100      | [HuggingFace README]                 | [fbnetc_100.rmsp_in1k]                  |

The model is downloaded from [HuggingFace][FBNet Model C100] at runtime and a custom performance optimization patch for [Intel® Data Center GPU Flex Series] is automatically applied. This optimization replaces the models default BatchNormAct2d operation - nn.functional.bach_norm - which uses pytorch's backend implementation instead of IPEX with nn.BatchNorm2d which is supported by IPEX. This patch is applied in the file [loader_utils](loader_utils.py). Its impact on performance when running with CUDA GPUs has not been verified.

# Dataset

> [!NOTE]
> Throughput and latency benchmarking can be done with dummy data (`./run_model.sh --dummy`). In such a case dataset setup can be skipped. As a downside expect to see low accuracy on the dummy data.

> [!NOTE]
> ~13.3 GB of free disk space is required to download and extract ImageNet dataset.

[ImageNet] validation dataset is required to measure accuracy during inference. Visit [ImageNet] site and download the following files:
* [ILSVRC2012_img_val.tar]
* [ILSVRC2012_devkit_t12.tar.gz]

> [!NOTE]
> Both dataset components must be downloaded to the same folder. This folder must be the `$DATASET_DIR` referenced in the following sections.

[get_dataset.sh] script can be used to download these files. There is no need to extract and format these files before running this sample. On the first run sample script will extract the archive with [torchvision.datasets.ImageNet]. Consequent runs will skip extraction.

# Prerequisites

Hardware:
* [Intel® Data Center GPU Flex Series]

Software:
* Intel® Data Center GPU Flex Series [Driver]
* [Intel® Extension for PyTorch]

# Run the model under container

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` under running container if you work behind the proxy.

Pull pre-built image with the sample:
```
docker pull intel/image-recognition:pytorch-flex-gpu-fbnet-inference
```
or build it locally:
```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/flex-gpu/pytorch-fbnet-inference/pytorch-flex-series-fbnet-inference.Dockerfile \
  -t intel/image-recognition:pytorch-flex-gpu-fbnet-inference .
```

Run sample as follows:
* With dummy data:

  * Running with dummy data is recommended for performance benchmarking (throughput and latency measurements)
  * Use higher `NUM_INPUTS` values for more precise peak performance results. `NUM_INPUTS` will be rounded to a multiple of `BATCH_SIZE`.
  * **NOTE**: Accuracy will be zero when using dummy data
  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  export BATCH_SIZE=1
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e MODEL_NAME=fbnetc_100 \
    -e PLATFORM=Flex \
    -e MAX_TEST_DURATION=60 \
    -e MIN_TEST_DURATION=60 \
    -e NUM_INPUTS=1000 \
    -e BATCH_SIZE=${BATCH_SIZE} \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    intel/image-recognition:pytorch-flex-gpu-fbnet-inference \
      /bin/bash -c "./run_model.sh --dummy"
  ```

* With ImageNet dataset (assumes that dataset was downloaded to the `$DATASET_DIR` folder):

  * Running with dataset images is recommended for accuracy measurements
  * In this mode, the test duration can be controlled by using the `NUM_INPUTS` parameter.
    The app tests a number of batches equal to `max(1, NUM_INPUTS // BATCH_SIZE)`
  * **NOTE**: Performance results (throughput and latency measurements) may be impacted due to data handling overhead
  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  export BATCH_SIZE=1
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e MODEL_NAME=fbnetc_100 \
    -e PLATFORM=Flex \
    -e NUM_INPUTS=50000 \
    -e BATCH_SIZE=${BATCH_SIZE} \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    -e DATASET_DIR=/dataset \
    -v $DATASET_DIR:/dataset \
    intel/image-recognition:pytorch-flex-gpu-fbnet-inference \
      /bin/bash -c "./run_model.sh"
  ```

Mind the following `docker run` arguments:

* HTTPS proxy is required to download model over network (`-e https_proxy=<...>`)
* `--cap-add SYS_NICE` is required for `numactl`
* `--device /dev/dri` is required to expose GPU device to running container
* `--ipc=host` is required for multi-stream benchmarking (`./run_model.sh --dummy --streams 2`) or large dataset cases
* `-v $DATASET_DIR:/dataset` in case where dataset is used. `$DATASET_DIR` should be replaced with the actual path to the ImageNet dataset.

# Run the model on baremetal

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` before running `run_model.sh` if you work behind proxy.

1. Download the sample:
   ```
   git clone https://github.com/IntelAI/models.git
   cd models/models_v2/pytorch/fbnet/inference/gpu
   ```
1. Create virtual environment `venv` and activate it:
   ```
   python3 -m venv venv
   . ./venv/bin/activate
   ```
1. Install sample python dependencies:
    ```
    python3 -m pip install -r requirements.txt
    ```
1. Install [Intel® Extension for PyTorch]
1. Add path to common python modules in the repo:
   ```
   export PYTHONPATH=$(pwd)/../../../../common
   ```
1. Setup required environment variables and run the sample with `./run_model.sh`:

   * With dummy data:

     * Running with dummy data is recommended for performance benchmarking (throughput and latency measurements)
     * Use higher `NUM_INPUTS` values for more precise peak performance results. `NUM_INPUTS` will be rounded to a multiple of `BATCH_SIZE`.
     * **NOTE**: Accuracy will be zero when using dummy data
     ```
     export MODEL_NAME=fbnetc_100
     export PLATFORM=Flex
     export BATCH_SIZE=1
     export MAX_TEST_DURATION=60
     export MIN_TEST_DURATION=60
     export NUM_INPUTS=1000
     export OUTPUT_DIR=/tmp/output
     ./run_model.sh --dummy
     ```
  * With ImageNet dataset (assumes that dataset was downloaded to the `$DATASET_DIR` folder):

    * Running with dataset images is recommended for accuracy measurements
    * In this mode, the test duration can be controlled by using the `NUM_INPUTS` parameter.
      The app tests a number of batches equal to `max(1, NUM_INPUTS // BATCH_SIZE)`
    * **NOTE**: Performance results (throughput and latency measurements) may be impacted due to data handling overhead
    ```
    export MODEL_NAME=fbnetc_100
    export PLATFORM=Flex
    export BATCH_SIZE=1
    export NUM_INPUTS=50000
    export OUTPUT_DIR=/tmp/output
    export DATASET_DIR=$DATASET_DIR
    ./run_model.sh
    ```

# Runtime arguments and environment variables

`run_model.sh` accepts a number of arguments to tune behavior. `run_model.sh` supports the use of environment variables as well as command line arguments for specifying these arguments (see the table below for details).

Before running `run_model.sh` script, user is required to:

* Set `OUTPUT_DIR` environment variable (or use `--output-dir`) where script should write logs.
* Use `--dummy` data or set `DATASET_DIR` environment variable (or use `--data`) pointing to ImageNet dataset.

Other arguments and/or environment variables are optional and should be used according to the actual needs (see examples above).

| Argument              | Environment variable | Valid Values      | Purpose                                                                                                                                  |
| --------------------- | -------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--ipex`              | `IPEX`               | `yes`             | Use [Intel® Extension for Pytorch] for XPU support (default: `yes`)                                                                      |
|                       |                      | `no`              | Use PyTorch XPU backend instead of [Intel® Extension for Pytorch]. Requires PyTorch version 2.4.0a or later.                             |
| `--amp`               | `AMP`                | `yes`             | Use AMP on model conversion to the desired precision (default: `yes`)                                                                    |
|                       |                      | `no`              |                                                                                                                                          |
| `--arch`              | `MODEL_NAME`         | `fbnetc_100`      | HuggingFace model to run (default: `fbnetc_100`)                                                                                         |
| `--batch-size`        | `BATCH_SIZE`         | >=1               | Batch size to use (default: `1`)                                                                                                         |
| `--data`              | `DATASET_DIR`        | String            | Location to load images from                                                                                                             |
| `--dummy`             | `DUMMY`              |                   | Use randomly generated dummy dataset in place of `--data` argument                                                                       |
| `--jit`               | `JIT`                | `none`            | JIT method to use (default: `trace`)                                                                                                     |
|                       |                      | `compile`         |                                                                                                                                          |
|                       |                      | `script`          |                                                                                                                                          |
|                       |                      | `trace`           |                                                                                                                                          |
| `--load`              | `LOAD_PATH`          |                   | Local path to load model from (default: disabled)                                                                                        |
| `--max-test-duration` | `MAX_TEST_DURATION`  | >=0               | Maximum duration in seconds to run benchmark. Testing will be truncated once maximum test duration has been reached. (default: disabled) |
| `--min-test-duration` | `MIN_TEST_DURATION`  | >=0               | Minimum duration in seconds to run benchmark. Images will be repeated until minimum test duration has been reached. (default: disabled)  |
| `--num-inputs`        | `NUM_INPUTS`         | >=1               | Number of images to load (default: `1`)                                                                                                  |
| `--output-dir`        | `OUTPUT_DIR`         | String            | Location to write output                                                                                                                 |
| `--proxy`             | `https_proxy`        | String            | System proxy                                                                                                                             |
| `--precision`         | `PRECISION`          | `bp16`            | Precision to use for the model (default: `fp32`)                                                                                         |
|                       |                      | `fp16`            |                                                                                                                                          |
|                       |                      | `fp32`            |                                                                                                                                          |
| `--save`              | `SAVE_PATH`          |                   | Local path to save model to (default: disabled)                                                                                          |
| `--streams`           | `STREAMS`            | >=1               | Number of parallel streams to do inference on (default: `1`)                                                                             |
| `--socket`            | `SOCKET`             | String            | Socket to control telemetry capture (default: disabled)                                                                                  |

> [!NOTE]
> * If `--dummy` is not specified (i.e. Quality Check mode), `--min/max-test-duration` settings are ignored. Test length is limited by minimum of num-inputs and the size of the dataset.

For more details, check help with `run_model.sh --help`

## Example output

Script output is written to the console as well as to the output directory in the file `output.log`.

For multi-stream cases per-stream results can be found in the `results_[STREAM_INSTANCE].json` files.

Final results of the inference run can be found in `results.yaml` file. More verbose results summaries are in `results.json` file.

The yaml file contents will look like:
```
results:
 - key: throughput
   value: 9199.48
   unit: img/s
 - key: latency
   value: 31.394199
   unit: ms
 - key: accuracy
   value: 76.06
   unit: percents
```

## Performance Benchmarking

[benchmark.sh] script can be used to benchmark FBNet performance for the [predefined use cases](profiles/README.md). The [benchmark.sh] script is a tiny FBNet specific wrapper on top of [benchmark.py](/models_v2/common/benchmark.py) script. The workflow for running a benchmark is as follows:

* (optional) Specify path to [svr-info](https://github.com/intel/svr-info):
  ```
  export PATH_TO_SVR_INFO=/path/to/svrinfo
  ```

* Specify path to output benchmark results (folder must be creatable/writable under `root`):
  ```
  export OUTPUT_DIR=/opt/output
  ```

* Run the benchmark script (assumes ``intel/image-recognition:pytorch-flex-gpu-fbnet-inference`` has already been pulled or built locally):
  ```
  sudo \
    PATH=$PATH_TO_SVR_INFO:$PATH \
    IMAGE=intel/image-recognition:pytorch-flex-gpu-fbnet-inference \
    OUTPUT_DIR=$OUTPUT_DIR \
    PROFILE=$(pwd)/models_v2/pytorch/fbnet/inference/gpu/profiles/c100.bf16.csv \
    PYTHONPATH=$(pwd)/models_v2/common \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^//') \
      $(pwd)/models_v2/pytorch/fbnet/inference/gpu/benchmark.sh
  ```

* Final output will be written to ``$OUTPUT_DIR``.

> [!NOTE]
> Additonal arguments that arent specified in the benchmark profile (``b0.bf16.csv`` in the example above) can be specified through environment variables as described in previous sections.

## Usage With CUDA GPU

Scripts have a matching degree of functionality for usage on CUDA GPU's. However, this is significantly less validated and so may not work as smoothly. The primary difference for using these scripts with CUDA is building the associated docker image. We will not cover CUDA on baremetal here. In addition Intel does not provide pre-built dockers for CUDA. These must be built locally.

```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/cuda-gpu/pytorch-fbnet-inference/pytorch-cuda-series-fbnet-inference.Dockerfile \
  -t intel/image-recognition:pytorch-cuda-gpu-fbnet-inference .
```

All other usage outlined in this README should be identical, with the exception of referencing this CUDA docker image in place of the for Intel GPU when running `docker run` as well as needing to add the `--gpus all` argument.

Example usage with dummy data is shown below:

```
mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
export BATCH_SIZE=1
docker run -it --rm --gpus all --ipc=host \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
  --cap-add SYS_NICE \
  --device /dev/dri/ \
  -e MODEL_NAME=fbnetc_100 \
  -e PLATFORM=CUDA \
  -e NUM_ITERATIONS=32 \
  -e NUM_IMAGES=${BATCH_SIZE} \
  -e BATCH_SIZE=${BATCH_SIZE} \
  -e OUTPUT_DIR=/tmp/output \
  -v /tmp/output:/tmp/output \
  intel/image-recognition:pytorch-cuda-gpu-fbnet-inference \
    /bin/bash -c "./run_model.sh --dummy"
```
