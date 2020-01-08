# Wide & Deep

This document has instructions for how to run Wide & Deep for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Store path to current directory and then clone `tensorflow/models`
       
    ```
    # We are going to use an older version of the tensorflow model repo.
    $ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
    $ pushd $MODEL_WORK_DIR

    $ git clone https://github.com/tensorflow/models.git tf_models
    $ cd tf_models
    $ git checkout 6ff0a53f81439d807a78f8ba828deaea3aaaf269 
    ```
    
2. Download and extract the pre-trained model.
    ```
    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_5/wide_deep_fp32_pretrained_model.tar.gz
    $ tar -xzvf wide_deep_fp32_pretrained_model.tar.gz
    ```
 
3. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

    ```
    $ cd $MODEL_WORK_DIR
    $ git clone https://github.com/IntelAI/models.git
    ```
4. Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running 
   following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/master/official/wide_deep/census_dataset.py). 
   Please note that below program requires `requests` module to be installed. You can install is using `pip install requests`. 
   Dataset will be downloaded in directory provided using `--data_dir`. If you are behind proxy then you can proxy urls 
   using `--http_proxy` and `--https_proxy` arguments.
   ```
   $ cd models
   $ python benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir $MODEL_WORK_DIR/widedeep_dataset
   ```

5. How to run

   * Running the model in online inference mode, set `--batch-size` = `1`
       ``` 
       $ cd $MODEL_WORK_DIR/models/benchmarks
    
       $ python launch_benchmark.py \
            --framework tensorflow \
            --model-source-dir $MODEL_WORK_DIR/tf_models \
            --precision fp32 \
            --mode inference \
            --model-name wide_deep \
            --batch-size 1 \
            --data-location $MODEL_WORK_DIR/widedeep_dataset \
            --checkpoint $MODEL_WORK_DIR/tf_models/wide_deep_fp32_pretrained_model \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-15 \
            --verbose
       ```
   * Running the model in batch inference mode, set `--batch-size` = `1024`
       ``` 
       $ cd $MODEL_WORK_DIR/models/benchmarks
    
       $ python launch_benchmark.py \
            --framework tensorflow \
            --model-source-dir $MODEL_WORK_DIR/tf_models \
            --precision fp32 \
            --mode inference \
            --model-name wide_deep \
            --batch-size 1024 \
            --data-location $MODEL_WORK_DIR/widedeep_dataset \
            --checkpoint $MODEL_WORK_DIR/tf_models/wide_deep_fp32_pretrained_model \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-15 \
            --verbose
       ```
6. The log file is saved to the value of `--output-dir`.

   The tail of the log output when the script completes should look
   something like this:

    ```
    accuracy: 1.0
    accuracy_baseline: 1.0
    auc: 1.0
    auc_precision_recall: 0.0
    average_loss: 2.1470942e-05
    global_step: 9775
    label/mean: 0.0
    loss: 2.1470942e-05
    precision: 0.0
    prediction/mean: 2.1461743e-05
    recall: 0.0
    End-to-End duration is %s 36.5971579552
    Latency is: %s 0.00224784460139
    current path: /workspace/benchmarks
    search path: /workspace/benchmarks/*/tensorflow/wide_deep/inference/fp32/model_init.py
    Using model init: /workspace/benchmarks/classification/tensorflow/wide_deep/inference/fp32/model_init.py
    PYTHONPATH: :/workspace/models
    RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --model-name=wide_deep         --precision=fp32         --mode=inference         --model-source-dir=/workspace/tf_models         --intelai-models=/workspace/intelai_models         --batch-size=1                  --data-location=/dataset         --checkpoint=/checkpoints
    ```

7. To return to where you started from:
```
$ popd
```