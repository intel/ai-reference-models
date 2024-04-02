# YOLO V5 Inference

YOLO V5 Inference best known configurations with [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

## Model Information

[get_model.sh]: get_model.sh
[run_model.sh]: run_model.sh

This sample uses source code and weights from the reference Pytorch implementation by its authors taken at the following commit to drive the inference:
* https://github.com/ultralytics/yolov5/commit/781401ec

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
  * Use higher `NUM_ITERATIONS` and lower `NUM_IMAGES` values (e.g. use `NUM_IMAGES=$BATCH_SIZE`) for more precise performance results
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
    -e PLATFORM=PLATFORM \
    -e NUM_ITERATIONS=32 \
    -e NUM_IMAGES=${BATCH_SIZE} \
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
    -e PLATFORM=PLATFORM \
    -e NUM_ITERATIONS=1 \
    -e NUM_IMAGES=5000 \
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
    python3 -m pip install -r requirements.txt
    ```
1. Install [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)
1. Add path to common python modules in the repo:
   ```
   export PYTHONPATH=$(pwd)/../../../../common
   ```
1. Setup required environment variables and run the sample with `./run_model.sh`:

   * With dummy data:

     * Running with dummy data is recommended for performance benchmarking (throughput and latency measurements)
     * Use higher `NUM_ITERATIONS` and lower `NUM_IMAGES` values (e.g. use `NUM_IMAGES=$BATCH_SIZE`) for more precise performance results
     * **NOTE**: Accuracy will be zero when using dummy data
     ```
     export MODEL_NAME=yolov5m
     export PLATFORM=Flex  # can also be: Max
     export BATCH_SIZE=1
     export NUM_ITERATIONS=32
     export NUM_IMAGES=${BATCH_SIZE}
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
     export NUM_ITERATIONS=1
     export NUM_IMAGES=50000
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

| Argument           | Environment variable | Valid Values      | Purpose                                                               |
| ------------------ | -------------------- | ----------------- | --------------------------------------------------------------------- |
| `--amp`            | `AMP`                | `no`              | Use AMP on model convertion to the desired precision (default: `no`)  |
|                    |                      | `yes`             |                                                                       |
| `--batch-size`     | `BATCH_SIZE`         | >=1               | Batch size to use (default: `1`)                                      |
| `--data`           | `DATASET_DIR`        | String            | Location to load images from                                          |
| `--dummy`          | `DUMMY`              |                   | Use randomly generated dummy dataset in place of `--data` argument    |
| `--jit`            | `JIT`                | `none`            | JIT method to use (default: `trace`)                                  |
| `--load`           | `LOAD_PATH`          |                   | Local path to load model from (default: disabled)                     |
|                    |                      | `trace`           |                                                                       |
|                    |                      | `script`          |                                                                       |
| `--num-images`     | `NUM_IMAGES`         | >=1               | Number of images to load (default: `1`)                               |
| `--num-iterations` | `NUM_ITERATIONS`     | >=1               | Number of times to test each batch (default: `100`)                   |
| `--output-dir`     | `OUTPUT_DIR`         | String            | Location to write output                                              |
| `--proxy`          | `https_proxy`        | String            | System proxy                                                          |
| `--precision`      | `PRECISION`          | `fp16`            | Precision to use for the model (default: `fp32`)                      |
|                    |                      | `fp32`            |                                                                       |
| `--save`           | `SAVE_PATH`          |                   | Local path to save model to (default: disabled)                       |
| `--streams`        | `STREAMS`            | >=1               | Number of parallel streams to do inference on (default: `1`)          |
| `--platform`       | `PLATFORM`           | `Flex`            | Platform that inference is being ran on                               |
|                    |                      | `Max`             |                                                                       |


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
