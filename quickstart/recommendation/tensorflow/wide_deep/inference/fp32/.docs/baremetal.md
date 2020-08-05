<!--- 50. Bare Metal -->
### Bare Metal

1. Download and untar the Wide & Deep FP32 inference model package:

    ```
    wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/wide-deep-fp32-inference.tar.gz
    tar -xvf wide-deep-fp32-inference.tar.gz
    ```

2. Clone `tensorflow/models` as a `tensorflow-models`
       
    ```
    # We going to use a branch based on older version of the tensorflow model repo.
    # Since, we need to to use logs utils on that branch, which were removed from 
    # the latest master
    $ git clone https://github.com/tensorflow/models.git tensorflow-models
    $ cd tensorflow-models
    $ git fetch origin pull/7461/head:wide-deep-tf2  
    $ git checkout wide-deep-tf2 
    ```

3. Once your environment is setup, navigate back to the directory that contains the Wide & Deep FP32 inference
   model package, set environment variables pointing to your dataset and output directories, and then run
   a quickstart script. To run inference with performance metrics:

    ```
    DATASET_DIR=<path to the Wide & Deep dataset directory>
    OUTPUT_DIR=<directory where log files will be written>
    TF_MODEL_SOURCE_DIR=<path to tensorflow-models>

    quickstart/fp32_inference_online.sh
    ```

