# Optimizations for Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*

## Overview

This document has instruction for running Intel® Extension for PyTorch* (IPEX) for
GPU in container.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 476.14](https://dgpu-docs.intel.com/releases/stable_476_14_20221021.html)
| Software | Docker* Installed |

## Get Started

### Installing the Intel Extensions for PyTorch
#### Docker pull command:

`docker pull intel/intel-extension-for-pytorch:xpu-flex`

### Running container:

Run following commands to start IPEX GPU tools container. You can use `-v` option to mount your
local directory into container. The `-v` argument can be omitted if you do not need
access to a local directory in the container. Pass the video and render groups to your
docker container so that the GPU is accessible.
```
IMAGE_NAME=intel/intel-extension-for-pytorch:xpu-flex
DOCKER_ARGS=${DOCKER_ARGS:---rm -it}

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run --rm \
    -v <your-local-dir>:/workspace \
    --group-add ${VIDEO} \
    ${RENDER_GROUP} \
    --device=/dev/dri \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    ${DOCKER_ARGS} \
    ${IMAGE_NAME} \
    bash
```

#### Verify if XPU is accessible from PyTorch:
You are inside container now. Run following command to verify XPU is visible to PyTorch:
```
python -c "import torch;print(torch.device('xpu'))"
```
Sample output looks like below:
```
xpu
```
Then, verify that the XPU device is available to IPEX:
```
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"
```
Sample output looks like below:
```
True
```
Finally, use the following command to check whether MKL is enabled as default:
```
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"
```
Sample output looks like below:
```
True
```

## Summary and Next Steps
Now you are inside container with Python 3.9, PyTorch and IPEX preinstalled. You can run your own script
to run on Intel GPU.

## Documentation and Sources

[GitHub* Repository](https://github.com/intel/intel-extension-for-pytorch/tree/master/docker)


## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.