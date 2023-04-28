## Docker

### Build the container

Extract the `<package name>` package and then run the `build.sh` script
to build the `<docker image>` container:

```
tar -xzf <package name>
cd <package dir>
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
    -it <docker image> /bin/bash
```

Once you're in the container, Run the following command to verify that you're able to import tensorflow
and check tensorflow version:

```
python -c "import tensorflow as tf; print(tf.__version__)"
```

After verifying that TensorFlow is available, you can run your own
script in the container.
