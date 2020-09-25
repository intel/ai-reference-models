<!--- 60. Docker -->
## Docker

The model container <docker image> includes the scripts and libraries needed to run 
<model name> <precision> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the output directory.

```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/fp32_inference.sh
```
