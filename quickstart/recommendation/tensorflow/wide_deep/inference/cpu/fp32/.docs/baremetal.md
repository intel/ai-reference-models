<!--- 50. Bare Metal -->
### Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

1. Download and untar the Wide & Deep FP32 inference model package:

    ```
    wget <package url>
    tar -xvf <package name>
    ```

2. Clone `tensorflow/models` as a `tensorflow-models`
       
    ```
    # We going to use a branch based on older version of the tensorflow model repo.
    # Since, we need to to use logs utils on that branch, which were removed from 
    # the latest master
    git clone https://github.com/tensorflow/models.git tensorflow-models
    cd tensorflow-models
    git fetch origin pull/7461/head:wide-deep-tf2
    git checkout wide-deep-tf2
    ```

3. Once your environment is setup, navigate back to the directory that contains the <model name> <precision> <mode>
   model package, set environment variables pointing to your dataset and output directories, and then run
   a quickstart script.
    ```
    DATASET_DIR=<path to the Wide & Deep dataset directory>
    OUTPUT_DIR=<directory where log files will be written>
    TF_MODELS_DIR=<path to tensorflow-models>
    # For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
    export BATCH_SIZE=<customized batch size value>

    ./quickstart/<script name>.sh
    ```
