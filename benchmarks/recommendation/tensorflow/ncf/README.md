## Neural Collaborative Filtering (NCF) ##

The following links have instructions on how to run NCF for the
following modes/precisions:
* [FP32 training](#training-instructions)
* [BFloat16 training](#training-instructions)
* [FP32 inference](/benchmarks/recommendation/tensorflow/ncf/inference/fp32/README.md)

## Training Instructions

1. Dataset

Support two datasets: ml-1m, ml-20m. It can be specified with flag `dataset=ml-1m` or `dataset=ml-20m`.
This model uses official tensorflow models repo, where [ncf](https://github.com/tensorflow/models/tree/master/official/recommendation)
model automatically downloads movielens ml-1m dataset as default if the `--data-location` flag is not set.
If you want to download movielens 1M/20M dataset and provide that path to `--data-location`, check this [reference](https://grouplens.org/datasets/movielens/)

2. Clone the official `tensorflow/models` repository.

For training, please checkout with tag `r2.1_model_reference `:
```
git clone https://github.com/tensorflow/models.git
cd models
git checkout r2.1_model_reference
```

3. Now clone `IntelAI/models` repository, then navigate to the `benchmarks` folder:

```
cd $MODEL_WORK_DIR
git clone https://github.com/IntelAI/models.git
cd models/benchmarks
```

4. Download and extract the pre-trained model, be careful it only works for 1M dataset.
Skip this step if training only.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_5/ncf_fp32_pretrained_model.tar.gz
tar -xzvf ncf_fp32_pretrained_model.tar.gz
```

5. Run the `launch_benchmark.py` script with the appropriate parameters.
* `--model-source-dir` - Path to official tensorflow models from step2.
* `--checkpoint` - Path to checkpoint directory for the Pre-trained model from step4. Checkpoint will be stored in this directory while training.

For training, suggest options are:
* `--batch-size 98304`
* `--precision fp32` or `--precision bfloat16`
* `dataset=ml-20m` - suggest to use 20M dataset for training
* `clean=1` - delete any files stored in `--checkpoint`. Disable this flag if want to reuse pre-trained model.
* `te=12` - set the max epoch. NCF will train 6+ epochs to SOTA, this flag will stop training when reach specific epoch. Set it to 1 if only want to test performance

```
python launch_benchmark.py \
    --checkpoint /home/<user>/ncf_fp32_pretrained_model \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ncf \
    --framework tensorflow \
    --mode training \
    --precision bfloat16 \
    --batch-size 98304 \
    --num-inter-threads 2 \
    --verbose \
    --docker-image intel/intel-optimized-tensorflow:1.15.2 \
    -- dataset=ml-20m clean=1 te=12
```

NCF will train 6+ epochs to SOTA. The tail of training log looks as below if trained to SOTA.
HR: Hit Ratio (HR), should >= 0.635 if use 20M dataset
NDCG: Normalized Discounted Cumulative Gain
```
I0122 12:00:14.874159 140303790921536 ncf_estimator_main.py:179] Iteration 6: HR = 0.6356, NDCG = 0.3787, Loss = 0.1567
I0122 12:00:14.874222 140303790921536 model_helpers.py:53] Stop threshold of 0.635 was passed with metric value 0.635591685772.
I0122 12:00:14.874658 140303790921536 mlperf_log.py:136] NCF_RAW_:::MLPv0.5.0 ncf 1579665614.874648094 (ncf_estimator_main.py:187) run_stop: {"success": true}
NCF_RAW_:::MLPv0.5.0 ncf 1579665614.881932020 (ncf_estimator_main.py:193) run_final
I0122 12:00:14.881944 140303790921536 mlperf_log.py:134] NCF_RAW_:::MLPv0.5.0 ncf 1579665614.881932020 (ncf_estimator_main.py:193) run_final
```

For training, suggest options are:
* `--batch-size 98304`
* `--precision fp32` or `--precision bfloat16`
* `dataset=ml-20m` - suggest to use 20M dataset for training
* `clean=1` - delete any files stored in `--checkpoint`. Disable this flag if want to reuse pre-trained model.
* `te=12` - set the max epoch. NCF will train 6+ epochs to SOTA, this flag will stop training when reach specific epoch. Set it to 1 if only want to test performance

```
python launch_benchmark.py \
    --checkpoint /home/<user>/ncf_fp32_pretrained_model \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name ncf \
    --framework tensorflow \
    --mode training \
    --precision bfloat16 \
    --batch-size 98304 \
    --num-inter-threads 2 \
    --verbose \
    --docker-image intel/intel-optimized-tensorflow:1.15.2 \
    -- dataset=ml-20m clean=1 te=12
```

NCF will train 6+ epochs to SOTA. The tail of training log looks as below if trained to SOTA.
HR: Hit Ratio (HR), should >= 0.635 if use 20M dataset
NDCG: Normalized Discounted Cumulative Gain
```
I0122 12:00:14.874159 140303790921536 ncf_estimator_main.py:179] Iteration 6: HR = 0.6356, NDCG = 0.3787, Loss = 0.1567
I0122 12:00:14.874222 140303790921536 model_helpers.py:53] Stop threshold of 0.635 was passed with metric value 0.635591685772.
I0122 12:00:14.874658 140303790921536 mlperf_log.py:136] NCF_RAW_:::MLPv0.5.0 ncf 1579665614.874648094 (ncf_estimator_main.py:187) run_stop: {"success": true}
NCF_RAW_:::MLPv0.5.0 ncf 1579665614.881932020 (ncf_estimator_main.py:193) run_final
I0122 12:00:14.881944 140303790921536 mlperf_log.py:134] NCF_RAW_:::MLPv0.5.0 ncf 1579665614.881932020 (ncf_estimator_main.py:193) run_final
```

6. To return to where you started from:
```
popd
```
