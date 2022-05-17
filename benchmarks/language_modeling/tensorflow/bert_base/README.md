# Bert

This document has instructions for how to run BERT Base Classifier model on MRPC, including training, frozen graph generating and inference based on the frozen graph.

## Prepare dataset

Currently, it supports MRPC(Microsoft Research Paraphrase Corpus) dataset on Classifier task. We need download it first. The [script](https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3) supports to download [GLUE](https://gluebenchmark.com/tasks) data set. You should download the script first and run command as below,

``` shell
WORKSPACE=${HOME}/bert_bf16
GLUE_DIR=${WORKSPACE}/data/glue
wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
python3 download_glue_data.py --data_dir $GLUE_DIR
```

## Prepare pre-trained model

This task supports multiple Bert Base model. You need to download one of them. For instance,

``` shell
BERT_BASE_DIR=${WORKSPACE}/pre-trained/uncased_L-12_H-768_A-12
cd $BERT_BASE_DIR && wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip -j uncased_L-12_H-768_A-12.zip
cd -
```

## Docker supports

If it's ran in docker environments, you should set the arguments of `--docker-image` and relevant `--volume` to `launch_benchmark.py` like below,

``` shell
--docker-image intel/intel-optimized-tensorflow:latest \
--volume $BERT_BASE_DIR:$BERT_BASE_DIR \
--volume $GLUE_DIR:$GLUE_DIR \
```

Of cause you can replace the docker image with yours which has TensorFlow environments.

## Training

1. Set environments variable
   ``` shell
   OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/bert_bf16/output/classifier}

   PYTHON=python3
   ```
2. Run the command to train. It supports double precision, fp32 and bfloat16. For instance, the fp32 training should be as below. The workspace must be in `benchmark` of [models](https://github.com/IntelAI/models).

   ``` shell
   ${PYTHON} launch_benchmark.py \
             --model-name=bert_base \
             --precision=fp32 \
             --mode=training \
             --framework=tensorflow \
             --batch-size=32 \
             --output-dir=${OUTPUT_DIR} \
             --verbose \
             -- \
             train-option=Classifier \
             task-name=MRPC \
             do-train=true \
             do-eval=true \
             data-dir=$GLUE_DIR/MRPC \
             vocab-file=$BERT_BASE_DIR/vocab.txt \
             config-file=$BERT_BASE_DIR/bert_config.json \
             init-checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
             max-seq-length=128 \
             learning-rate=2e-5 \
             num-train-epochs=3 \
             do-lower-case=True \
             num-intra-threads=28 \
             num-inter-threads=1 \
             experimental_mkldnn_ops=True \
             do_lower_case=True

   ```

It will store the checkpoint in the `$OUTPUT_DIR`.

## Export to frozen graph

After the end of training, you need to export the checkpoint to frozen graph first. By default, it will export frozen graph. You can set `--saved_model=true` to export the saved model too. For example, to export a model with precision fp32 should be as below. At the end, it will generate the frozen graph named `frozen_graph.pb` in `$OUTPUT_DIR/frozen`.

``` shell
${PYTHON} export_classifier.py \
          --task_name=MRPC \
          --bert_config_file=$BERT_BASE_DIR/bert_config.json \
          --output_dir=${OUTPUT_DIR} \
          --precision=fp32 \
          --saved_model=true \
          --experimental_gelu=True # Disable this flag if your TenorFlow doesn't support
```

## Inference

To run inference with frozen graph, we first switch to a different inference output dir as below,

``` shell
OUTPUT_DIR=${${WORKSPACE}/output/classifier_frozen}
FROZEN_DIR=${${WORKSPACE}/output/classifier/frozen}
```

Now we can run the inference. It supports two precisions too. For example, the fp32 mode inference should be as below,

``` shell
${PYTHON} launch_benchmark.py \
        --model-name=bert_base \
        --precision=fp32 \
        --mode=inference \
        --framework=tensorflow \
        --batch-size=1 \
        --output-dir=${OUTPUT_DIR} \
        --in-graph=${FROZEN_DIR}/frozen_graph.pb \
        --socket-id=0 \
        --verbose \
        -- \
        infer-option=Classifier \
        task-name=MRPC \
        do-eval=true \
        data-dir=$GLUE_DIR/MRPC \
        vocab-file=$BERT_BASE_DIR/vocab.txt \
        config-file=$BERT_BASE_DIR/bert_config.json \
        max-seq-length=128 \
        do-lower-case=True \
        profile=False \
        num-intra-threads=24 \
        num-inter-threads=1
```
