# Deep Interest Evolution Network for Click-Through Rate Prediction

### This code for this model is originally from: [Alibaba DIEN repo](https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN_TF2)

This document has instructions for how to run [DIEN](https://arxiv.org/abs/1809.03672) for the
following modes/platforms:
* [FP32 training](#fp32-training)
* [FP32 inference](#fp32-inference)
* [BFLOAT16 inference](#bfloat16-inference)


## FP32 Training

### 1. Prepare datasets
```
export DATASET_DIR=/path/to/dien-dataset-folder

# download datasets
wget https://zenodo.org/record/3463683/files/data.tar.gz
wget https://zenodo.org/record/3463683/files/data1.tar.gz
wget https://zenodo.org/record/3463683/files/data2.tar.gz

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```

### 2. Run training
Please specify the `data-location`.
```
python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --model-name dien \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --socket-id 0 \
    --batch-size 128 \
    --docker-image intel/intel-optimized-tensorflow:latest
```

Below is a sample log file tail when training:
```
approximate_accelerator_time: 196.536
iter: 4000 ----> train_loss: 1.3396 ---- train_accuracy: 0.7166 ---- train_aux_loss: 1.0679
save model iter: 4000
iteration:  4000
iter: 4000
Total recommendations: 512000
Approximate accelerator time in seconds is 196.536
Approximate accelerator performance in recommendations/second is 2605.117
Ran training with batch size 128
Log file location: {--output-dir value}/benchmark_dien_training_fp32_20201118_100251.log
```

## FP32 Inference
>Note: If you run on Windows systems, please use a browser to download the dataset (step 1) and the pretrained model files (step 2).
For Linux systems, please use the following code snippets.
### 1. Prepare datasets
```
export DATASET_DIR=/path/to/dien-dataset-folder

# download datasets
wget https://zenodo.org/record/3463683/files/data.tar.gz
wget https://zenodo.org/record/3463683/files/data1.tar.gz
wget https://zenodo.org/record/3463683/files/data2.tar.gz

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
### 2. Prepare pretrained model
```
export PB_DIR=/path/to/dien-pretrained-folder
# download frozen pb(s)
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_fp32_static_rnn_graph.pb
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_fp32_pretrained_opt_model.pb
```

### Run FP32 inference on Linux:
* **Throughput**:
    Please specify the `data-location` and `in-graph`. 
    Note that --num-intra-threads and --num-inter-threads need to be specified depending on the requirement/machine.
    Please specify graph_type as `static` if you are using static RNN graph along with `dien_fp32_static_rnn_graph.pb`
    ```
    python launch_benchmark.py \
        --data-location $DATASET_DIR \
        --in-graph $PB_DIR/dien_fp32_static_rnn_graph.pb \
        --model-name dien \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --batch-size 128 \
        --num-intra-threads 26 \
        --num-inter-threads 1 \
        --graph_type=static \
        --exact-max-length=100 \
        --docker-image intel/intel-optimized-tensorflow:latest
    ```
    
    or `dynamic` if using dynamic RNN graph along with `dien_fp32_pretrained_opt_model.pb`
    ```
    python launch_benchmark.py \
        --data-location $DATASET_DIR \
        --in-graph $PB_DIR/dien_fp32_pretrained_opt_model.pb \
        --model-name dien \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --batch-size 128 \
        --num-intra-threads 26 \
        --num-inter-threads 1 \
        --graph_type=dynamic \
        --docker-image intel/intel-optimized-tensorflow:latest \
    ```
    
    Output is as below. Performance is reported as recommendations/second
    ```
    Max length :100
    test_auc: 0.8375 ---- test_accuracy: 0.754075370 ---- eval_time: 4.137
    test_auc: 0.8375 ---- test_accuracy: 0.754075370 ---- eval_time: 4.124
    num_iters  947
    batch_size  128
    niters  2
    Total recommendations: 121216
    Approximate accelerator time in seconds is 4.127
    Approximate accelerator performance in recommendations/second is 29345.576
    ```
* **Accuracy**:
    Please specify the `data-location` and `in-graph`.
    ```
    python launch_benchmark.py \
        --data-location $DATASET_DIR \
        --in-graph $PB_DIR/dien_fp32_static_rnn_graph.pb \
        --model-name dien \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --batch-size 128 \
        --num-intra-threads 26 \
        --num-inter-threads 1 \
        --accuracy-only \
        --exact-max-length=100 \
        --docker-image intel/intel-optimized-tensorflow:latest
    ```
    
    Below is a sample log file tail when testing accuracy:
    
    ```
    test_auc: 0.8375 ---- test_accuracy: 0.754075370 
    
    Ran inference with batch size 128
    ```
    
* **Latency**:
    Please specify the `data-location` and `in-graph`.
    To check for latency set the batch-size to 1 and check 
    the time taken. Since the dataset for DIEN has varying 
    sequential length, an additional option to set sequential
    length can be used. The option is ```--exact-max-length.```
    Another option is ```num-iterations```. This options can be used
    to run inference multiple times to get average performance over 
    the num of iterations specified.
    
    Please specify graph_type as `static` if you are using static RNN graph along with `dien_fp32_static_rnn_graph.pb` 
    ```
    python launch_benchmark.py \
        --data-location $DATASET_DIR \
        --in-graph $PB_DIR/dien_fp32_static_rnn_graph.pb \
        --model-name dien \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --batch-size 1 \
        --num-intra-threads 26 \
        --num-inter-threads 1 \
        --graph_type=static \
        --exact-max-length=100 \
        --docker-image intel/intel-optimized-tensorflow:latest \
        -- num-iterations=10
    ```
    
    or `dynamic` if using dynamic RNN graph along with `dien_fp32_pretrained_opt_model.pb`
    ```
    python launch_benchmark.py \
        --data-location $DATASET_DIR \
        --in-graph $PB_DIR/dien_fp32_pretrained_opt_model.pb \
        --model-name dien \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --batch-size 1 \
        --num-intra-threads 26 \
        --num-inter-threads 1 \
        --graph_type=dynamic \
        --exact-max-length=100 \
        --docker-image intel/intel-optimized-tensorflow:latest \
        -- num-iterations=10
    ```
    
    Since DIEN is not a big model checking for latency for batch-size 1
    may show a much lower throughput 
    Below is a sample log file tail when testing latency for max length 100:
    ```
    Exact Max length set to : 100
    test_auc: 0.8172 ---- test_accuracy: 0.679653680 ---- eval_time: 12.991
    test_auc: 0.8172 ---- test_accuracy: 0.679653680 ---- eval_time: 12.995
    num_iters  1848
    batch_size  1
    niters  2
    Total recommendations: 1848
    Approximate accelerator time in seconds is 12.994
    Approximate accelerator performance in recommendations/second is 142.231
    ```

### Run FP32 inference on Windows:
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/tensorflow/Windows.md).
Then, run inference to calculate throughput, accuracy, or latency.
* **Throughput**:
    Please specify the `data-location` and `in-graph`. 
    Note that --num-intra-threads and --num-inter-threads need to be specified depending on the requirement/machine.
    Please specify graph_type as `static` if you are using static RNN graph along with `dien_fp32_static_rnn_graph.pb`.
    Using `cmd.exe`, run:
    ```
    python launch_benchmark.py ^
        --data-location <path to dataset directory> ^
        --in-graph <path to pretrained model>\\dien_fp32_static_rnn_graph.pb ^
        --model-name dien ^
        --framework tensorflow ^
        --precision fp32 ^
        --mode inference ^
        --batch-size 128 ^
        --num-intra-threads 26 ^
        --num-inter-threads 1 ^
        --graph_type=static ^
        --exact-max-length=100
    ```
    
    or `dynamic` if using dynamic RNN graph along with `dien_fp32_pretrained_opt_model.pb`
    ```
    python launch_benchmark.py ^
        --data-location <path to dataset directory> ^
        --in-graph <path to pretrained model>\\dien_fp32_pretrained_opt_model.pb ^
        --model-name dien ^
        --framework tensorflow ^
        --precision fp32 ^
        --mode inference ^
        --batch-size 128 ^
        --num-intra-threads 26 ^
        --num-inter-threads 1 ^
        --graph_type=dynamic
    ```
* **Accuracy**:
    Please specify the `data-location` and `in-graph`. Using `cmd.exe`, run:
    ```
    python launch_benchmark.py ^
        --data-location <path to dataset directory> ^
        --in-graph <path to pretrained model>\\dien_fp32_static_rnn_graph.pb ^
        --model-name dien ^
        --framework tensorflow ^
        --precision fp32 ^
        --mode inference ^
        --batch-size 128 ^
        --num-intra-threads 26 ^
        --num-inter-threads 1 ^
        --accuracy-only ^
        --exact-max-length=100
    ```  
* **Latency**:
    Please specify the `data-location` and `in-graph`.
    To check for latency set the batch-size to 1 and check 
    the time taken. Since the dataset for DIEN has varying 
    sequential length, an additional option to set sequential
    length can be used. The option is ```--exact-max-length.```
    Another option is ```num-iterations```. This options can be used
    to run inference multiple times to get average performance over 
    the num of iterations specified.
    
    Please specify graph_type as `static` if you are using static RNN graph along with `dien_fp32_static_rnn_graph.pb`.
    Using `cmd.exe`, run: 
    ```
    python launch_benchmark.py ^
        --data-location <path to dataset directory> ^
        --in-graph <path to pretrained model>\\dien_fp32_static_rnn_graph.pb ^
        --model-name dien ^
        --framework tensorflow ^
        --precision fp32 ^
        --mode inference ^
        --batch-size 1 ^
        --num-intra-threads 26 ^
        --num-inter-threads 1 ^
        --graph_type=static ^
        --exact-max-length=100 ^
        -- num-iterations=10
    ```
    
    or `dynamic` if using dynamic RNN graph along with `dien_fp32_pretrained_opt_model.pb`
    ```
    python launch_benchmark.py ^
        --data-location <path to dataset directory> ^
        --in-graph <path to pretrained model>\\dien_fp32_pretrained_opt_model.pb ^
        --model-name dien ^
        --framework tensorflow ^
        --precision fp32 ^
        --mode inference ^
        --batch-size 1 ^
        --num-intra-threads 26 ^
        --num-inter-threads 1 ^
        --graph_type=dynamic ^
        --exact-max-length=100 ^
        -- num-iterations=10
    ```
    Since DIEN is not a big model checking for latency for batch-size 1
    may show a much lower throughput.

## BFLOAT16  Inference
>Note: If you run on Windows systems, please use a browser to download the dataset (step 1) and the pretrained model files (step 2).
For Linux systems, please use the following code snippets.
### 1. Prepare dataset as in FP32 instructions

### 2. Download pretrained bfloat16 model file

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_bf16_pretrained_opt_model.pb

### 3. Run inference with precision set to bfloat16 for throughput, accuracy and latency 
       
  Same as fp32 except change the precision to bfloat16. All output log tails are similar
  to those generated for fp32

```
python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --in-graph $PB_DIR/dien_bf16_pretrained_opt_model.pb \
    --model-name dien \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --batch-size 128 \
    --num-intra-threads 26 \
    --num-inter-threads 1 \
    --graph_type=dynamic \
    --exact-max-length=100 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- num-iterations=10
```

Below is a sample log file tail when testing throughput:
```
test_auc: 0.8301 ---- test_accuracy: 0.750915721 ---- eval_time: 15.685
num_iters  947
batch_size  128
niters  1
Total recommendations: 121216
Approximate accelerator time in seconds is 15.685
Approximate accelerator performance in recommendations/second is 7728.245
Ran inference with batch size 128
Log file location: {--output-dir value}/benchmark_dien_inference_fp32_20201118_094143.log
```

To run inference with bfloat16 for throughput, accuracy or latency on Linux or Windows systems,
use same options as FP32 in sections [Run FP32 inference on Linux](#run-fp32-inference-on-linux) & [Run FP32 inference on Windows](#run-fp32-inference-on-windows), and use precision as bfloat16.
 