<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset (where train.csv and eval.csv are located), an output directory where log
files will be written, and optionally a directory where checkpoint files can
be read and written from.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset directory>
export OUTPUT_DIR=<directory where the logs and the saved model will be written>
export CHECKPOINT_DIR=<Optional directory where checkpoint files will be read and written>
```

Train the model (The model will be trained for 10 epochs if -- steps is not specified)
```
python launch_benchmark.py \
  --model-name wide_deep_large_ds \
  --precision fp32 \
  --mode training  \
  --framework tensorflow \
  --batch-size 512 \
  --data-location ${DATASET_DIR} \
  --checkpoint ${CHECKPOINT_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --docker-image <docker image>
```
Once the training completes successfully the path of checkpoint files and saved_model.pb will be printed as shown below
```
INFO:tensorflow:SavedModel written to: $OUTPUT_DIR/temp-1602670603/saved_model.pb
Using TensorFlow version 2.4.0
Begin training and evaluation
Saving model checkpoints to $CHECKPOINT_DIR
****Computing statistics of train dataset*****
estimator built
fit done
evaluate done
Model exported to $OUTPUT_DIR
```
