## Neural Collaborative Filtering (NCF) ##

This document has instructions for how to run NCF for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference.

## FP32 Inference Instructions

1. Dataset

This model uses official tensorflow models repo, where [ncf](https://github.com/tensorflow/models/tree/master/official/recommendation)
model automatically downloads movielens ml-1m dataset as default if the `--data-location` flag is not set.
If you want to download movielens 1M dataset and provide that path to `--data-location`, check this [reference](https://grouplens.org/datasets/movielens/1m/)

2. Clone the official `tensorflow/models` repository with  tag `v1.11` and make a small change to `data_async_generation.py`, commenting out a line that causes a crash in the model script. Store the path to the current directory.

```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR

$ git clone https://github.com/tensorflow/models.git tf_models
$ cd tf_models
$ git checkout v1.11
$ sed -i.bak 's/atexit.register/# atexit.register/g' official/recommendation/data_async_generation.py
```

3. Now clone `IntelAI/models` repository, then navigate to the `benchmarks` folder:

```
$ cd $MODEL_WORK_DIR
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

4. Download and extract the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_5/ncf_fp32_pretrained_model.tar.gz
$ tar -xzvf ncf_fp32_pretrained_model.tar.gz
```

5. Run the `launch_benchmark.py` script with the appropriate parameters.
* `--model-source-dir` - Path to official tensorflow models from step2.
* `--checkpoint` - Path to checkpoint directory for the Pre-trained model from step4


For batch inference, `--batch-size 256`, `--socket-id 0`, `--checkpoint` path from step5, `--model-source-dir` path from step2

```
$ python launch_benchmark.py \
    --checkpoint $MODEL_WORK_DIR/models/benchmarks/ncf_trained_movielens_1m \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intel/intel-optimized-tensorflow:1.15.2
```

The tail of batch inference log, looks as below.
```
...
2018-11-12 19:42:44.851050: step 22900, 931259.2 recommendations/sec, 0.27490 msec/batch
2018-11-12 19:42:44.880778: step 23000, 855571.2 recommendations/sec, 0.29922 msec/batch
2018-11-12 19:42:44.910551: step 23100, 870836.8 recommendations/sec, 0.29397 msec/batch
2018-11-12 19:42:44.940675: sE1112 19:42:45.420336 140101437536000 tf_logging.py:110] CRITICAL - Iteration 1: HR = 0.2248, NDCG = 0.1132
tep 23200, 867319.7 recommendations/sec, 0.29516 msec/batch
2018-11-12 19:42:44.971828: step 23300, 867319.7 recommendations/sec, 0.29516 msec/batch
2018-11-12 19:42:45.002699: step 23400, 861751.1 recommendations/sec, 0.29707 msec/batch
2018-11-12 19:42:45.033635: step 23500, 873671.1 recommendations/sec, 0.29302 msec/batch
Average recommendations/sec across 23594 steps: 903932.8 (0.28381 msec/batch)
...
```

For online inference, `--batch-size 1`, `--socket-id 0`, `--checkpoint` path from step5, `--model-source-dir` path from step2

```
$ python launch_benchmark.py \
    --checkpoint $MODEL_WORK_DIR/models/benchmarks/ncf_trained_movielens_1m \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 1 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intel/intel-optimized-tensorflow:1.15.2
```

The tail of online inference log, looks as below.
```
...
2018-11-12 20:24:25.175342: step 6039900, 4675.9 recommendations/sec, 0.21386 msec/batch
2018-11-12 20:24:25.198717: step 6040000, 4905.6 recommendations/sec, 0.20385 msec/batch
Average recommendations/sec across 6040001 steps: 4573.0 (0.21920 msec/batch)
...
```
For Accuracy, `--batch-size 256`, `--socket-id 0`, `--checkpoint` path from step5, `--model-source-dir` path from step2

```
$ python launch_benchmark.py \
    --checkpoint $MODEL_WORK_DIR/models/benchmarks/ncf_trained_movielens_1m \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name ncf \
    --socket-id 0 \
    --accuracy-only \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intel/intel-optimized-tensorflow:1.15.2
```

The tail of accuracy log, looks as below.
HR: Hit Ratio (HR)
NDCG: Normalized Discounted Cumulative Gain
```
...
E0104 20:03:50.940653 140470332344064 tf_logging.py:110] CRITICAL - Iteration 1: HR = 0.2290, NDCG = 0.1148
...
```

6. To return to where you started from:
```
$ popd
```
