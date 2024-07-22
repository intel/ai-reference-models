# IFRNet Model Inference Sample

[IFRNet]: https://arxiv.org/abs/2205.14620
[Intel® Extension for Pytorch]: https://intel.github.io/intel-extension-for-pytorch/index.html#installation
[IFRNet Model]: https://github.com/ltkong218/IFRNet
[Intel® Data Center GPU Flex Series]: https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html
[Driver]: https://dgpu-docs.intel.com/driver/installation.html
[Vimeo-90K]: http://toflow.csail.mit.edu/
[vimeo_interp_test.zip]: http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
[get_dataset.sh]: get_dataset.sh
[get_model.sh]: get_model.sh
[run_model.sh]: run_model.sh
[patches]: patches

IFRNet is an encoder-decoder based network to accomplish video frame interpolation. This directory contains a sample implementation of image interpolation based on [IFRNet] inference. It is targeted to run on Intel Discrete Graphics platforms (XPUs) by leveraging [Intel® Extension for Pytorch].

The sample supports two modes of execution:
* A performance benchmarking mode where the sample executes IFRNet inference based on dummy tensor initialization for a specified duration or number of `inputs` or `frames`. The `input` is a tensor capturing a pair of input images that is fed to IFRNet inference. The output is tensor representing the interpolated frame.
* A quality-check mode which takes in frames from the Vimeo-90K Triplet Test dataset, measures the quality (in PSNR) of the interpolated frames against references contained in the dataset. A percentage score of passing frames is reported, and if specified to do so, the output frames are dumped as well.

The rest of this document covers more details about the model, dataset, and the control knobs for each mode of execution. Further, instructions are provided on how to use the scripts in this directory for execution in bare-metal and docker container environments.

# Model and Sources

This sample uses source code and weights from the reference Pytorch implementation by its authors ([IFRNet Model], cited below) to drive the inference.

    @InProceedings{Kong_2022_CVPR, 
      author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie}, 
      title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation}, 
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year = {2022}
    }

The weights and model need to be downloaded from the original implementation and installed under `checkpoints` and `models` directories respectively for the sample to be usable. (Options for customizing these are discussed [later in this document](#Runtime-arguments-and-environment-variables)). Further, it is recommended that the model is patched with a custom performance fix, provided in the [patches] directory.

The path for the patched model must be specified on the PYTHONPATH env variable to be usable by the application.

The [get_model.sh] script may be used to accomplish all of this. The model run script ([run_model.sh]) is configured to invoke this script and automatically install the  `./models` directory if not found. The default (and simplest) invocation is as below.
```
./get_model.sh
```
For optional arguments to use a non-default model path, please refer to the [get_model.sh] script.

# Dataset

To run a test in quality-check mode, the dataset must be fetched and moved to the required path (that will be specified to the application command line).

> [!NOTE]
> Throughtput and latency benchmarking can be done with dummy data by specifying performance mode of execution (`./run_model.sh --dummy`). For such a run, quality assessment is skipped.

The [Vimeo-90K] dataset is leveraged for the quality-check mode execution. Specifically, the Triplet Test subset of this large dataset is used for this. The archive [vimeo_interp_test.zip] on the [Vimeo-90K] website covers this.

More info and the relevant citation are shared below:

    @article{xue17toflow,
      author = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
      title = {Video Enhancement with Task-Oriented Flow},
      journal = {arXiv},
      year = {2017}
    }

> [!NOTE]
> A minimum ~6 GB of free disk space is required to download and extract Vimeo-90K Triplet Test dataset. Note that the setup script installs the dataset regardless of which mode is executed following it.

[get_dataset.sh] script can be used to download these files into the current working directory as below. Note that the script will skip download and extraction, if it finds a file/directory of the same name as the target tarball or the directory capturing the extracted dataset.
```
./get_dataset.sh
```

## Dataset Resolution and Performance Impacts

The interpolated frames will have resolutions matching the provided dataset. In the case of dummy data which is randomly generated this can be controlled through the command line arguments ``--data-channels``, ``--data-height``, ``--data-width`` or the equivalent environment variables ``DATA_CHANNELS``, ``DATA_HEIGHT``, ``DATA_WIDTH``.

> [!NOTE]
> The selected dataset resolution has a direct impact on the resulting throughput. Default resolution in dummy mode is 1280x720p with 3-channels. Mind that Vimeo-90K dataset uses 448x256 resolution which will have higher throughput than the default dummy mode.

# Prerequisites

Hardware:
* [Intel® Data Center GPU Flex Series]

Software:
* Intel® Data Center GPU Flex Series [Driver]
* [Intel® Extension for PyTorch]

# Running the model in a docker container

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` under running container if you work behind the proxy.

Pull pre-built image with the sample:
```
docker pull intel/image-interpolation:pytorch-flex-gpu-ifrnet
```
or build it locally:
```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/flex-gpu/pytorch-ifrnet-inference/pytorch-flex-series-ifrnet-inference.Dockerfile \
  -t intel/image-interpolation:pytorch-flex-gpu-ifrnet .
```

## Performance Benchmarking mode

To run the sample in performance mode, start `run_model.sh` with `--dummy` command line option. A sample command line that mounts an output directory from host machine into the docker container is provided below.

  * The sample provides the same dummy input tensors to all inference submissions for the duration of the run. The output is not evaluated (**NOTE**: quality report will be zero when using performance mode).
  * It is recommended to control the test duration by using the controls `MIN_TEST_DURATION` and `MAX_TEST_DURATION` in this mode.
  * Note that there is no `BATCH_SIZE` parameter supported in current implementation, and this is equivalent to running with `BATCH_SIZE=1`
  * Output files capturing the performance metrics are dumped into the specified output directory. 
  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e PLATFORM=Flex \
    -e MIN_TEST_DURATION=60 \
    -e MAX_TEST_DURATION=60 \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    intel/image-interpolation:pytorch-flex-gpu-ifrnet \
      /bin/bash -c "./run_model.sh --dummy"
  ```
## Quality Check mode

In this mode the sample runs with Vimeo-90K Triplet Test dataset (assumes that dataset was downloaded to the `$DATASET_DIR` folder):

  * Running with the dataset images is recommended to get a quality report
  * In this mode, the test duration can be controlled by using the `NUM_INPUTS` parameter. The app tests a number of inputs equal to `min(NUM_INPUTS, length of dataset`)
  * Use higher `NUM_INPUTS` (e.g. 20000 for full dataset) to cover a larger range of inputs for more reliable quality reports
  * **NOTE**: Performance results (throughput and latency measurements) will likely be sub-par compared to the benchmarking mode due to data handling overhead

  ```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  docker run -it --rm --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e PLATFORM=Flex \
    -e NUM_INPUTS=3000 \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    -e DATASET_DIR=/dataset \
    -v $DATASET_DIR:/dataset \
    intel/image-interpolation:pytorch-flex-gpu-ifrnet \
      /bin/bash -c "./run_model.sh"
  ```

Mind the following `docker run` arguments:

* HTTPS proxy is required to download model over network (`-e https_proxy=<...>`)
* `--cap-add SYS_NICE` is required for `numactl`
* `--device /dev/dri` is required to expose GPU device to running container
* `--ipc=host` is required for multi-process execution and synchronization
* `-v $DATASET_DIR:/dataset` in case where dataset is used. `$DATASET_DIR` should be replaced with the actual path to the Vimeo-90K dataset. This can be installed through [Dataset Instructions](#dataset).
* `-e OUTPUT_DIR=/tmp/output` specifies the output directory for the run, which in turn is mounted from the host by `-v /tmp/output:/tmp/output` 

# Run the model on baremetal

> [!NOTE]
> Sample requires network connection to download model from the network via HTTPS. Make sure to set `https_proxy` before running `run_model.sh` if you work behind proxy.

1. Download the sample:
   ```
   git clone https://github.com/IntelAI/models.git
   cd models/models_v2/pytorch/ifrnet/inference/gpu
   ```
1. Create virtual environment `venv` and activate it:
   ```
   python3 -m venv venv
   . ./venv/bin/activate
   ```
1. Install required dependencies and dataset:
    ```
    ./setup.sh
    ```
1. Install [Intel® Extension for PyTorch]
1. Add path to common python modules in the repo:
   ```
   export PYTHONPATH=$(pwd)/../../../../common
   ```
1. Setup required environment variables and run the sample with `./run_model.sh`

## Performance Benchmarking mode

To run the sample in performance mode, set env variable `DUMMY` env variable set to `"yes"`. A sample command line is provided below

  * The sample provides the same dummy input tensors to all inference submissions for the duration of the run. The output is not evaluated.
  * It is recommended to control the test duration by using the controls `MIN_TEST_DURATION` and `MAX_TEST_DURATION` in this mode.
  * Note that there is no `BATCH_SIZE` parameter supported in current implementation, and this is equivalent to running with `BATCH_SIZE=1`
  * Output files capturing the performance metrics are dumped into the specified output directory. 
  * **NOTE**: Quality report will be zero when using performance mode

     ```
     export PLATFORM=Flex
     export OUTPUT_DIR=/tmp/output
     export MIN_TEST_DURATION=60
     export MAX_TEST_DURATION=60
     ./run_model.sh --dummy
     ```
## Quality Check mode

In this mode the sample is run with Vimeo-90K Triplet Test dataset. Following instructions assume the dataset is available at the `$DATASET_DIR` folder.  This can be fetched through [Dataset Instructions](#dataset).

  * Running with the dataset images is recommended to get a quality report
  * In this mode, the test duration can be controlled by using the `NUM_INPUTS` parameter. The app tests a number of inputs equal to `min(NUM_INPUTS, length of dataset`)
  * Use higher `NUM_INPUTS` (e.g. 20000 for full dataset) to cover a larger range of inputs for more reliable quality reports
  * **NOTE**: Performance results (throughput and latency measurements) will likely be sub-par compared to the benchmarking mode due to data handling overhead

    ```
    export PLATFORM=Flex
    export NUM_INPUTS=3000
    export OUTPUT_DIR=/tmp/output
    export DATASET_DIR=$DATASET_DIR
    ./run_model.sh
    ```

# Runtime arguments and environment variables

`run_model.sh` accepts a number of arguments to tune behavior. `run_model.sh` supports the use of environment variables as well as command line arguments for specifying these arguments (see the table below for details).

Before running `run_model.sh` script, user is required to:

* Set `OUTPUT_DIR` environment variable (or use `--output-dir`) where script should write logs.
* Use `--dummy` or set `DATASET_DIR` environment variable (or use `--data`) pointing to the dataset.

Other arguments and/or environment variables are optional and should be used according to the actual needs (see examples above).
For more details, check help with `run_model.sh --help`


| Argument                          | Environment variable | Valid Values      | Purpose                                                               |
| --------------------------------- | -------------------- | ----------------- | --------------------------------------------------------------------- |
| `--ipex`                          | `IPEX`               | `yes`             | Use [Intel® Extension for Pytorch] for XPU support (default: `yes`)   |
|                                   |                      | `no`              | Use PyTorch XPU backend instead of [Intel® Extension for Pytorch]. Requires PyTorch version 2.4.0a or later. |
| `--data`                          | `DATASET_DIR`        | String            | Location to load images from                                          |
| `--modelsdir`                     | `MODELS_DIR`         | String            | Location to read model from (default: ./models)                       |
| `--dummy`                         | `DUMMY`              |                   | Use randomly generated dummy dataset in place of `--data` argument    |
| `--data-channels`                 | `DATA_CHANNELS`      | Integer           | Number of color channels of randomly generated dataset (default: 3)   |
| `--data-height`                   | `DATA_HEIGHT`        | Integer           | Height of images in randomly generated dataset (default: 720)         |
| `--data-width`                    | `DATA_WIDTH`         | Integer           | Width of images in randomly generated dataset (default: 1280)         |
| `--pretrained-weights`            | `LOAD_PATH`          | String            | Local path to load model from (default: [IFRNet_Vimeo90K.pth])        |
| `--num-inputs`                    | `NUM_INPUTS`         | >=1               | Max pairs of input images to test (default: `100`). See note below    |
| `--async`                         | `ASYNC`              | >=0               | Number of batches after which to issue a gpu sync. Default: 0(=all)   |
| `--precision`                     | `PRECISION`          | fp16/bf16/fp32    | Datatype for model and input tensors in inference (Default: fp16)     |
| `--amp`                           | `AMP`                |                   | Use Pytorch's Autocast for mixed precision (Default: disabled)        |
| `--streams`                       | `STREAMS`            | >=1               | Number of parallel processes/streams to run inference (Default: 1)    |
| `--interpolation`                 | `INTERPOLATION`      | Integer           | Socket to enable telemetry capture. Default="" (disabled)             |
| `--output-dir`                    | `OUTPUT_DIR`         | String            | Location to write output                                              |
| `--platform`                      | `PLATFORM`           | Flex/Max/cuda/cpu | Device to run the model on                                            |
| `--warmup`                        | `WARMUP        `     | Integer           | Number of frames to run as warmup for the inference model             |
| `--saveimages`                    | `SAVEIMAGES`         |                   | Save output images to output-dir                                      |
| `--psnr-threshold`                | `MIN_PSNR_DB`        | Integer           | Min PSNR in dB (default 25) for one frame to pass a quality check     |
| `--min-pass-pct`                  | `MIN_PASS_PCT`       | Integer           | Min % of frames (default 95) to pass to consider the full run a pass  |
| `--min-test-duration`             | `MIN_TEST_DURATION`  | Integer           | Min duration (in seconds) to run the test. See Note below.            |
| `--max-test-duration`             | `MAX_TEST_DURATION`  | Integer           | Max duration (in seconds) to run the test. See Note below.            |
| `--socket`                        | `SOCKET`             | String            | Socket to enable telemetry capture. Default="" (disabled)             |

> [!NOTE]
> * With `--dummy`, (i.e in Performance Benchmarking mode), `--min/max-test-duration` settings override `--num-inputs` setting.
> * If `--dummy` is not specified (i.e. Quality Check mode), `--min/max-test-duration` settings are ignored. Test length is limited by minimum of num-inputs and the size of the dataset.

The weights used in this sample by default correspond to the [Vimeo-90K] dataset, and are tuned for generating 1 interpolated frame per input frame pair.

## Example output

Script output is written to the console as well as to the output directory in the file `output.log`.

Final results of the inference run can be found in `results.yaml` file. More verbose results summaries are in `results.json` file.

The yaml file contents will look like:
```
results:
 - key: throughput
   value: 33.48
   unit: img/s
 - key: latency
   value: 31.39
   unit: ms
 - key: accuracy
   value: 96.55
   unit: percents
```

## Performance Benchmarking

[benchmark.sh] script can be used to benchmark IFRNet performance for the [predefined use cases](profiles/README.md). The [benchmark.sh] script is a tiny sample-specific wrapper on top of [benchmark.py](/models_v2/common/benchmark.py) script. The workflow for running a benchmark is as follows:

* (optional) Specify path to [svr-info](https://github.com/intel/svr-info):
  ```
  export PATH_TO_SVR_INFO=/path/to/svrinfo
  ```

* Specify path to output benchmark results (folder must be creatable/writable under `root`):
  ```
  export OUTPUT_DIR=/opt/output
  ```

* Run the benchmark script (assumes ``intel/image-interpolation:pytorch-flex-gpu-ifrnet`` has already been pulled or built locally):
  ```
  sudo \
    PATH=$PATH_TO_SVR_INFO:$PATH \
    IMAGE=intel/image-interpolation:pytorch-flex-gpu-ifrnet \
    OUTPUT_DIR=$OUTPUT_DIR \
    PROFILE=$(pwd)/models_v2/pytorch/IFRNet/inference/gpu/profiles/IFRNet.fp16.csv \
    PYTHONPATH=$(pwd)/models_v2/common \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^//') \
      $(pwd)/models_v2/pytorch/IFRNet/inference/gpu/benchmark.sh
  ```

* Final output will be written to ``$OUTPUT_DIR``.

> [!NOTE]
> Additonal arguments that arent specified in the benchmark profile (``IFRNet.fp16.csv`` in the example above) can be specified through environment variables as described in previous sections.

## Usage With CUDA GPU

Scripts have a matching degree of functionality for usage on CUDA GPU's. However, this is significantly less validated and so may not work as smoothly. The primary difference for using these scripts with CUDA is building the associated docker image. We will not cover CUDA on baremetal here. In addition Intel does not provide pre-built dockers for CUDA. These must be built locally.

```
docker build \
  $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
  -f docker/cuda-gpu/pytorch-ifrnet-inference/pytorch-cuda-series-ifrnet-inference.Dockerfile \
  -t intel/image-interpolation:pytorch-cuda-gpu-ifrnet .
```

All other usage outlined in this README should be identical, with the exception of referencing this CUDA docker image in place of the for Intel GPU when running `docker run` as well as needing to add the `--gpus all` argument.

Example usage with dummy data is shown below:

```
  mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  docker run -it --rm --gpus all --ipc=host \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    --cap-add SYS_NICE \
    --device /dev/dri/ \
    -e PLATFORM=CUDA \
    -e MIN_TEST_DURATION=60 \
    -e MAX_TEST_DURATION=60 \
    -e OUTPUT_DIR=/tmp/output \
    -v /tmp/output:/tmp/output \
    intel/image-interpolation:pytorch-cuda-gpu-ifrnet \
      /bin/bash -c "./run_model.sh --dummy"
```
