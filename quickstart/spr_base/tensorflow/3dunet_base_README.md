<!--- 0. Title -->
# Intel TensorFlow tools container

<!-- 10. Description -->
## Description

This document has instructions for building and running the Intel-optimized
TensorFlow container using the container package.

## Container Package

The container package includes wheel for TensorFlow, a Dockerfile, and a script to build the container.

```
tensorflow-spr-3dunet
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── tensorflow-spr-3dunet.Dockerfile
└── whls
    └── tf_nightly-2.9.0.202212-cp38-cp38-linux_x86_64.whl
```

## Docker

### Build the container

Extract the `tensorflow-spr-3dunet.tar.gz` package and then run the `build.sh` script
to build the `model-zoo:tensorflow-spr-3dunet` container:

```
tar -xzf tensorflow-spr-3dunet.tar.gz
cd tensorflow-spr-3dunet
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
    -it model-zoo:tensorflow-spr-3dunet /bin/bash
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

