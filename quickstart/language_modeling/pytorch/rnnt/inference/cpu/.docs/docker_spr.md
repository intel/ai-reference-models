## Run the model

Set the `CHECKPOINT_DIR` environment variable and run the script to download the
pretrained model:
```
export CHECKPOINT_DIR=<directory to download the pretrained model>
mkdir -p $CHECKPOINT_DIR
cd <package dir>
bash download_model.sh
```

After you've downloaded the pretrained model and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode> in
docker. Set environment variables to specify the dataset directory, precision to run,
and an output directory. By default, the `inference_realtime.sh` quickstart script will
be run. To run a different quickstart script, set the `SCRIPT` environment variable
to the script of your choice.

The snippet below demonstrates how to run <model name> <mode>:
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export CHECKPOINT_DIR=<path to the downloaded model weights directory>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Run a different quickstart script
SCRIPT=accuracy.sh ./run.sh
```
