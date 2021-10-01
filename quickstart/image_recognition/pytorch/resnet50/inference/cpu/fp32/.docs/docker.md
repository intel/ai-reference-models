<!--- 60. Docker -->
## Docker

Use the base [PyTorch 1.8 container](https://hub.docker.com/layers/intel/intel-optimized-pytorch/1.8.0/images/sha256-5ca5d619b33bc6abc42cef654e9ee119ed0959c65f37de22a0bd8764c71412dd?context=explore) 
`intel/intel-optimized-pytorch:1.8.0` to run <model name> <precision> <mode>.
To run the model quickstart scripts using the base PyTorch 1.8 container,
you will need to provide a volume mount for the <package dir> package.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the [ImageNet validation dataset](#dataset). The accuracy
script also downloads the pretrained model at runtime, so provide proxy
environment variables, if necessary.

```
DATASET_DIR=<path to the dataset folder>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume <path to the model package directory>:/<package dir> \
  --privileged --init -it \
  intel/intel-optimized-pytorch:1.8.0 /bin/bash
```

Synthetic data is used when running batch or online inference, so no
dataset mount is needed.

```
docker run \
  --privileged --init -it \
  --volume <path to the model package directory>:/<package dir> \
  intel/intel-optimized-pytorch:1.8.0 /bin/bash
```

Run quickstart scripts:
```
cd /<package dir>
bash quickstart/<script name>.sh
``` 

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
