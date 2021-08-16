<!--- 0. Title -->
# Intel TensorFlow tools container

<!-- 10. Description -->
## Description

This document has instructions for building and running the Intel-optimized
Tensorflow container using the container package.

## Container Package

The container package includes wheel for Tensorflow, a Dockerfile, and a script to build the container.

```
tensorflow-spr
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_v2.0_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.0_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.3_PyTorch.txt
│       └── licenses.txt
├── tensorflow-spr.Dockerfile
└── whls
    └── tf_nightly-2.6.0-cp36-cp36m-linux_x86_64.whl
```

## Docker

### Build the container

Extract the `tensorflow-spr.tar.gz` package and then run the `build.sh` script
to build the `tensorflow-ww22-spr:243fa30a-centos-8` container:

```
tar -xzf tensorflow-spr.tar.gz
cd tensorflow-spr
./build.sh
```

### Running the container

The following command can be used to interactively run the TensorFlow
container. You can optionally use the `-v` (or `--volume`) option to mount
your local directory into the container.
```
docker run \
    -v <your-local-dir>:/workspace \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -it tensorflow-ww22-spr:243fa30a-centos-8 /bin/bash
```

Once you're in the container, Run the following command to verify that you're able to import tensorflow
and check tensorflow version:

```
python -c "import tensorflow as tf; print(tf.__version__)"
```

After verifying that TensorFlow is available, you can run your own
script in the container.

<!--- 80. License -->
## License

Licenses can be found in the container package, in the `licenses` directory.

