## Build the container

The <model name> <mode> package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the <model name> <mode> container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf <package name>
cd <package dir>

# Build the container
./build.sh
```

After the build completes, you should have a container called
`<docker image>` that will be used to run the model.