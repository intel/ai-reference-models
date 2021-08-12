<!--- 0. Title -->
# PyTorch ResNet50 training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 training using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run ResNet50 training in a container.
```
pytorch-spr-resnet50-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_v2.4_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_PyTorch.txt
│       └── licenses.txt
├── model_packages
│   └── pytorch-spr-resnet50-training.tar.gz
├──pytorch-spr-resnet50-training.Dockerfile
└── run.sh
```

## Datasets

### ImageNet

Download and extract the ImageNet2012 training and validation dataset from
[http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` and `train` directories should be set as the
`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Build the container

The ResNet50 training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         fecc7096a11e        40 minutes ago      8.31GB
```

To build the ResNet50 training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-resnet50-training.tar.gz
cd pytorch-spr-resnet50-training

# Build the container
./build
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-resnet50-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run ResNet50 training in docker. Set environment variables to
specify the dataset directory, precision to run, and an output directory.

```
# Navigate to the container package directory
cd pytorch-spr-resnet50-training

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

