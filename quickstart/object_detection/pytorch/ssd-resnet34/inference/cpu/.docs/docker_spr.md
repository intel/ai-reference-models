## Run the model

Download the pretrained model weights using the script from the MLPerf repo
and set the `PRETRAINED_MODEL` environment variable to point to the downloaded file:
```
wget https://raw.githubusercontent.com/mlcommons/inference/v0.7/others/cloud/single_stage_detector/download_model.sh
sh download_model.sh
export PRETRAINED_MODEL=$(pwd)/pretrained/resnet34-ssd1200.pth
```

After downloading the pretrained model and following the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode>
using docker. Set environment variables to specify the dataset directory,
precision to run, and an output directory for logs. By default, the `run.sh`
script will run the `inference_realtime.sh` quickstart script. To run a different
script, specify the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export DATASET_DIR=<path to the coco dataset>
export PRETRAINED_MODEL=<path to the resnet34-ssd1200.pth file>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To run a difference quickstart script, us the SCRIPT env var
SCRIPT=accuracy.sh ./run.sh
```
