# Optimizations for Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instruction for running Tensorflow using Intel GPU in container.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed:[Download Driver 476.14](https://dgpu-docs.intel.com/releases/stable_476_14_20221021.html)
| Software | Docker* Installed |

## Get Started

### Installing the Intel Extensions for TensorFlow
#### Docker pull command:

`docker pull intel/intel-extension-for-tensorflow:gpu-flex`

#### Running container:

Run following commands to start TF GPU  tools container. You can use `-v` option to mount your
local directory into container. 

```
IMAGE_NAME=intel/intel-extension-for-tensorflow:gpu-flex
DOCKER_ARGS=${DOCKER_ARGS:---rm -it}

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run \
    -v <your-local-dir>:/workspace \
    --group-add ${VIDEO} \
    ${RENDER_GROUP} \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    ${DOCKER_ARGS} \
    ${IMAGE_NAME} \
    bash
```

##### Verify if GPU is accessible from Tensorflow:
You are inside container now. Run following command to verify GPU is visible to Tensorflow:

```
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
You should be able to see GPU device in list of devices. Sample output looks like below:

```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 9266936945121049176
xla_global_id: -1
, name: "/device:XPU:0"
device_type: "XPU"
locality {
bus_id: 1
}
incarnation: 15031084974591766410
physical_device_desc: "device: 0, name: INTEL_XPU, pci bus id: <undefined>"
xla_global_id: -1
, name: "/device:XPU:1"
device_type: "XPU"
locality {
bus_id: 1
}
incarnation: 17448926295332318308
physical_device_desc: "device: 1, name: INTEL_XPU, pci bus id: <undefined>"
xla_global_id: -1
]
``` 
## Documentation and Sources

[GitHub* Repository](https://github.com/intel/intel-extension-for-tensorflow/tree/main/docker)

## Summary and Next Steps

Now you are inside container with Python 3.9 and Tensorflow 2.10.0 preinstalled. You can run your own script
to run on intel GPU. 

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.