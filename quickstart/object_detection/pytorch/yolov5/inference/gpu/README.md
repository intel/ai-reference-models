<!--- 0. Title -->
# Yolov5 inference

<!-- 10. Description -->
## Description

This document has instructions for running Yolov5 inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series 170 or 140
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-pytorch/) to install the latest IPEX version and other prerequisites.

- Intel® oneAPI Base Toolkit: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Threading Building Blocks (oneTBB)
  - Intel® oneAPI Math Kernel Library (oneMKL)
  - Follow [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline) to download and install the latest oneAPI Base Toolkit.

  - Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```

<!--- 30. Datasets -->
## Datasets

Download and extract the 2017 training/validation images and annotations from the [COCO dataset website](https://cocodataset.org/#download) to a `coco` folder and unzip the files. After extracting the zip files, your dataset directory structure should look something like this:
```
coco
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   ├── 000000454854.jpg
│   ├── 000000137045.jpg
│   ├── 000000129582.jpg
│   └── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── 000000000632.jpg
    └── ...
```
The parent of the `annotations`, `train2017`, and `val2017` directory (in this example `coco`) is the directory that should be used when setting the `IMAGE_FILE` environment
variable for YOLOv5 (for example: `export IMAGE_FILE=/home/<user>/coco/val2017/000000581781.jpg`). In addition, we should also set the `size` environment to match the size of image.
(for example: `export size=416`)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Inference with FP16 and FP32 for specified batch size on Flex series 170 or 140 |

<!--- 50. Baremetal -->
## Run the model
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate to Model Zoo directory:
  ```bash
  cd models
  ```

### Run the model on Baremetal
Set environment variables to run the quickstart script:
```
export IMAGE_FILE=<path to coco dataset image>
export PRECISION=<provide fp16 precision>
export OUTPUT_DIR=<Path to save the output logs>
export GPU_TYPE=<provide either flex_170 or flex_140>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>
export NUM_ITERATIONS=<set number of iterations else it will run with default iterations>

Run the model specific dependencies:
NOTE: Installing dependencies in setup.sh may require root privilege
./quickstart/object_detection/pytorch/yolov5/inference/gpu/setup.sh

# Run a quickstart script
./quickstart/object_detection/pytorch/yolov5/inference/gpu/inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
