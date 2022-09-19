## Docker

### Build the container

Extract the `<package name>` package and then run the `build.sh` script
to build the <docker image> conatiner. Note that the size of the conatainer
after it's built is about 8.31GB. The container is based on CentOS 8.3.2011
and includes anaconda, PyTorch, Intel Extensions for PyTorch, MKL, and other
dependencies needed to run PyTorch models.
```
tar -xzf <package name>
cd <package dir>

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

python -c "import intel_extension_for_pytorch"
```

After verifying that PyTorch and IPEX are available, you can run your own
script in the container.
