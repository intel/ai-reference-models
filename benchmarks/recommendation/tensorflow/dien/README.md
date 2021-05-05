# Deep Interest Evolution Network for Click-Through Rate Prediction

### This code for this model is originally from: [Alibaba DIEN repo](https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN_TF2)

This document has instructions for how to run [DIEN](https://arxiv.org/abs/1809.03672) for the
following modes/platforms:
* [FP32 training](#fp32-training)
* [FP32 inference](#fp32-inference)
* [BFLOAT16 inference](#fp32-inference)


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
    --docker-image intel/intel-optimized-tensorflow:2.4.0
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
# download frozen pb
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/dien_fp32_pretrained_model.pb
```

### 3. Run inference  with fp32 for throughput 
Please specify the `data-location` and `in-graph`. 
Note that --num-intra-threads and --num-inter-threads need to be specified dpending on the requirement/machine.
```
python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --in-graph $PB_DIR/dien_fp32_pretrained_model.pb \
    --model-name dien \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 128 \
    --num-intra-threads 26 \
    --num-inter-threads 1  \
    --docker-image intel/intel-optimized-tensorflow:2.4.0
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
### 4. Run inference with fp32 precision for accuracy
Please specify the `data-location` and `in-graph`.
```
python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --in-graph $PB_DIR/dien_fp32_pretrained_model.pb \
    --model-name dien \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 128 \
    --num-intra-threads 26 \
    --num-inter-threads 1  \
    --accuracy_only  \
    --docker-image intel/intel-optimized-tensorflow:2.4.0
```

Below is a sample log file tail when testing accuracy:

```
test_auc: 0.8375 ---- test_accuracy: 0.754075370 

Ran inference with batch size 128
```

### 5. Run inference with fp32 precision for latency
Please specify the `data-location` and `in-graph`.
To check for latency set the batch-size to 1 and check 
the time taken. Since the dataset for DIEN has varying 
sequential length, an additional option to set sequential
length can be used. The option is ```--exact-max-length```
```
python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --in-graph $PB_DIR/dien_fp32_pretrained_model.pb \
    --model-name dien \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --num-intra-threads 26 \
    --num-inter-threads 1  \
    --exact-max-length 100 \
    --docker-image intel/intel-optimized-tensorflow:2.4.0
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
## BFLOAT16  Inference
### 1. Prepare dataset as in FP32 instructions

### 2. Download pretrained bfloat16 model file

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/dien_bf16_pretrained_model.pb

### 3. Run inference with precision set to bfloat16 for throughput, accuracy and latency 
       
  Same as fp32 except change the precision to bfloat16. All output log tails are similar
  to those generated for fp32

python launch_benchmark.py \
    --data-location $DATASET_DIR \
    --in-graph $PB_DIR/dien_bf16_pretrained_model.pb \
    --model-name dien \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --batch-size 128 \
    --num-intra-threads 26 \
    --num-inter-threads 1  \
    --docker-image intel/intel-optimized-tensorflow:2.4.0



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

### 4. Run inference  with bfloat16 for accuracy  or latency 
 Use same options as fp32 in section 4 & 5, and use precision as bfloat16.
