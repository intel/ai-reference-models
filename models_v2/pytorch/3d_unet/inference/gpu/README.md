# 3D-UNet Model Inference

3D U-Net is CNN for 3D biomedical image segmentation. See these two papers for more details: 
* [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
* [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z).


## Model and sources
The sample is based on [v1.7.1](https://github.com/MIC-DKFZ/nnUNet/tree/v1.7.1) of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de/) and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

 Model has been trained on "fold 1" part of the BraTS 2019 dataset and model is downloaded automatically 
by [setup script](setup.sh). To download it manually go to https://zenodo.org/records/3904106 and download
`fold_1.zip` file that holds model weights, training statistics, etc.

## Dataset
> [!NOTE]
> For dummy mode, dataset  is not necessary, and its download and setup may be skipped.

In this sample we show how to use model trained on Multimodal Brain Tumor Segmentation Challenge 2019 (BraTS 2019)
 dataset. See [here](https://www.med.upenn.edu/cbica/brats-2019/) for more details about this dataset.

This sample can be used to evaluate both performance and accuracy of the model. For performance mode 
 sample uses dummy data. Set `DUMMY` environment variable to `yes` to enable it. For dummy mode, dataset 
 is not necessary, and its download and setup may be skipped.
 
For accuracy evaluation, set `DUMMY` environment variable to `no`. For this mode, BraTS 2019 dataset
 should be downloaded. To do so, go to https://www.med.upenn.edu/cbica/brats-2019/, create account,
 request access to dataset, download `MICCAI_BraTS_2019_Data_Training.zip` file, unzip it and map folder to `/dataset` one inside the docker. See command line example below. Inference script will use this dataset to prepare data for accuracy check and store intermediate, aka preprocessed data in the same folder.
 
If automatic preprocessing is not possible, for example due to insufficient permissions during inference run, then
 it can be done manually. To run preprocessing manually perform all steps described in the "Run the model under container"
 or "Run the model on baremetal" sections of this readme but instead of running `./run_model.sh` run `./preprocess.sh`. It should take
 about one minute to finish preprocessing and after that inference script can be run with only `read` permissions for dataset folder. 
 
Accuracy check is performed for images listed in one of the `models_v2/pytorch/3d_unet/inference/gpu/folds/foldX_validation.txt`
 files. See setup script for more details how to select specific file. By default, `fold1_validation.txt` file is used.
 
In the dataset, images are provided in [NIfTI](https://nifti.nimh.nih.gov/) (Neuroimaging Informatics Technology Initiative) file format,
 and have to be preprocessed by inference script before used as a model input. Model accepts four channel PyTorch
 tensor as input [4, 224, 224, 160] and provides segmentation map of the same dimension as output.


## Prerequisites

Hardware:
* [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)

Software:
* [Intel® Data Center GPU Flex Series Driver](https://dgpu-docs.intel.com/driver/installation.html)
* [Intel® Extension for Pytorch](https://github.com/intel/intel-extension-for-pytorch)


## Run the model under container

Build docker image. 
```
docker build \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') \
    -f docker/flex-gpu/pytorch-3dunet-inference/pytorch-flex-series-3dunet-inference.Dockerfile \
    -t intel/image-recognition:pytorch-flex-gpu-3dunet-inference .
```

Run sample in performance mode with dummy dataset.
```
mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  
docker run --rm -it --device /dev/dri/ --cap-add SYS_NICE --ipc=host --shm-size=5g \
    -e PLATFORM=Flex \
    -e OUTPUT_DIR="/tmp/output" \
    -e DUMMY=yes \
    -e PRECISION=fp16 \
    -e BATCH_SIZE=1 \
    -e NUM_ITERATIONS=10 \
    -v /tmp/output:/tmp/output \
    intel/image-recognition:pytorch-flex-gpu-3dunet-inference \
    /bin/bash -c "./run_model.sh"
```

Run sample in accuracy mode. In this example, dataset has been downloaded to `/home/MICCAI_BraTS_2019_Data_Training/` folder on host.
```
mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
  
docker run --rm -it --device /dev/dri/ --cap-add SYS_NICE --ipc=host --shm-size=5g \
    -e PLATFORM=Flex \
    -e OUTPUT_DIR="/tmp/output" \
    -e DUMMY=no \
    -e PRECISION=fp16 \
    -e BATCH_SIZE=1 \
    -e NUM_ITERATIONS=10 \
    -e DATASET_DIR=/dataset \
    -v $DATASET_DIR:/dataset \
    -v /tmp/output:/tmp/output \
    intel/image-recognition:pytorch-flex-gpu-3dunet-inference \
    /bin/bash -c "./run_model.sh"
```

## Run the model on baremetal

1. Download the sample:
   ```
   git clone https://github.com/IntelAI/models.git
   cd models/models_v2/pytorch/3d_unet/inference/gpu
   ```
1. Create virtual environment and activate it:
   ```
   python3 -m venv venv
   . ./venv/bin/activate
   ```
1. Install sample python dependencies:
    ```
    python3 -m pip install -r requirements.txt
    ```
1. Install [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
1. Add path to common python modules in the repo:
   ```
   export PYTHONPATH=$(pwd)/../../../../common
   ```
1. Install model:
    ```
    ./setup.sh
    ```
1. Download and uzip dataset. Set path to unzipped folder. This step can be skipped for `dummy` mode.
   ```
   export DATASET_DIR="dataset_folder"
   ```
1. Run sample:

   * Run sample in performance mode with dummy dataset:
    ```
    mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
    export PLATFORM=Flex
    export OUTPUT_DIR="/tmp/output"
    export DUMMY=yes
    export PRECISION=fp16
    export BATCH_SIZE=1
    export NUM_ITERATIONS=10
    ./run_model.sh
    ```
     
   * Run sample in accuracy mode:  
    ```
    mkdir -p /tmp/output && rm -f /tmp/output/* && chmod -R 777 /tmp/output
    export PLATFORM=Flex
    export OUTPUT_DIR="/tmp/output"
    export DUMMY=no
    export PRECISION=fp16
    export BATCH_SIZE=1
    export NUM_ITERATIONS=10
    ./run_model.sh
    ```


## Runtime arguments

Runtime arguments can be passed as command line parameters or as environment variables. Table below summarizes environment variables used in the examples above.

|     Argument         | Environment variable  |  Valid Values         | Purpose                                                               |
|----------------------| --------------------- |-----------------------| --------------------------------------------------------------------- |
| `--ipex`             | `IPEX`                | `yes`                 | Use [Intel® Extension for Pytorch] for XPU support (default: `yes`)   |
| `--jit`              | `JIT`                 | `none`                | JIT method to use (default: `trace`)                                  |
|                      |                       | `compile`             |                                                                       |
|                      |                       | `trace`               |                                                                       |
| `--platform`         | `PLATFORM`            | `Flex`, `Max`, `CUDA`, `CPU` | Run on the device in the specified plarform group              |
| `--dummy`            | `DUMMY`               | `yes`, `no`           | If `yes`, run model on dummy data, dataset maybe absent. If `no`, use real data and perform accuracy check on model output. (default: `yes`)|
| `--precision`        | `PRECISION`           | `fp32`,`fp16`,`bf16`  | Datatype to use. (default: `fp16`)|
| `--batch-size`       | `BATCH_SIZE`          | 1, 2, ...             | Number of images in single inference call. Maximum supported value varies depending on datatype and available memory, but usually is small, 1 or 2 images. (default: `1`)|
| `--num-iterations`   | `NUM_ITERATIONS`      | 1, 2, ...             | Number of inference calls. Total number of processed images in one script call is `BATCH_SIZE * NUM_ITERATIONS`. (default: `10`)|
| `--amp`              | `AMP`                 | `yes`, `no`           | Use AMP on model conversion. (default: `yes`)|
| `--output-dir`       | `OUTPUT_DIR`          |                       | Location to write output to. |
| n/a                  | `DATASET_DIR`         |                       | Path to dataset. |
| `--multi-tile`       | `MULTI_TILE`          | `True`, `False`       | Run benchmark in multi-tile configuration. |



# Output example
Script output is written to the console as well as to the output directory. This is the accuracy test example from the console:
```
INFO[1/1]: PERF_STATUS[200/200] Throughput=2.22 img/s | Latency=450.85 ms | Total Accuracy=89.3336% | Whole Tumor Accuracy=90.4854% | Core Tumor Accuracy=91.4035% | Enhanced Tumor Accuracy=86.1120% | Throughput /w Overhead=2.17 img/s | Latency /w Overhead=460.00 ms
```
and from `results.yaml` file:
```
results:
 - key: throughput
   value: 2.223558942623231
   unit: img/s
 - key: latency
   value: 450.84524393081665
   unit: ms
 - key: accuracy
   value: 89.33363195948799
   unit: percents
```


