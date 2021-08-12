<!--- 0. Title -->
# Intel PyTorch Extension (IPEX) tools container

<!-- 10. Description -->
## Description

This document has instructions for building and running the Intel-optimized
PyTorch container using the container package. This container is used as the
base for the PyTorch model containers.

## Container Package

The container package includes wheels for PyTorch and the Intel PyTorch
Extension (IPEX), a Dockerfile, and a script to build the container.

```
pytorch-ipex-spr
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_v2.4_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_PyTorch.txt
│       └── licenses.txt
├── pytorch-ipex-spr.Dockerfile
└── whls
    ├── torch-1.10.0a0+git6f40371-cp37-cp37m-linux_x86_64.whl
    └── torch_ipex-1.1.0-cp37-cp37m-linux_x86_64.whl
```

## Docker

### Build the container

Extract the `pytorch-ipex-spr.tar.gz` package and then run the `build.sh` script
to build the model-zoo:pytorch-ipex-spr conatiner. Note that the size of the conatainer
after it's built is about 8.31GB. The container is based on CentOS 8.3.2011
and includes anaconda, PyTorch, Intel Extensions for PyTorch, MKL, and other
dependencies needed to run PyTorch models.
```
tar -xzf pytorch-ipex-spr.tar.gz
cd pytorch-ipex-spr

./build.sh
```

### Running the container

The following command can be used to interactively run the PyTorch/IPEX
container. You can optionally use the `-v` (or `--volume`) option to mount
your local directory into the container.
```
docker run \
    -v <your-local-dir>:/workspace \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -it model-zoo:pytorch-ipex-spr /bin/bash
```

Once you're in the container, activate the pytorch environment:
```
# List the available conda environments
conda env list

# Activate the pytorch environment
source activate pytorch
```

Run the following command to verify that you're able to import PyTorch and
IPEX from python:
```
python -c "import torch"

python -c "import intel_pytorch_extension"
```

> NOTE: It is expected that warnings will be displayed after importing the Intel
> PyTorch Extension (IPEX), due to redefining the autocast operation inside IPEX. The
> autocast inside IPEX has more functionalities such as: jit support, int8 support,
> verbose debug etc., compared to stock PyTorch. The warnings will not affect functionality
> or performance.

After verifying that PyTorch and IPEX are available, you can run your own
script in the container.

<!--- 80. License -->
## License

Licenses can be found in the container package, in the `licenses` directory.

