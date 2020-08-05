!<--- 60. Docker -->
### Docker

 When running in docker, the Wide & Deep FP32 inference container includes the model package and tensorflow model source repo,
   which is needed to run inference. To run the quickstart scripts, you'll need to provide volume mounts for the dataset and 
   an output directory where log files will be written.

    To run inference with performance metrics:
    ```
    DATASET_DIR=<path to the Wide & Deep dataset directory>
    OUTPUT_DIR=<directory where log files will be written>

    docker run \
    --env DATASET_DIR=${DATASET_DIR} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --volume ${DATASET_DIR}:${DATASET_DIR} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --privileged --init -t \
    amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-recommendation-wide-deep-fp32-inference \
    /bin/bash ./quickstart/fp32_inference_online.sh
    ```
