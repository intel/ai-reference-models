<!--- 50. Baremetal -->

## Run the model

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
```
# Navigate to the model zoo directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: fp32, bfloat16 or bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the quickstart scripts:
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training.sh
```
