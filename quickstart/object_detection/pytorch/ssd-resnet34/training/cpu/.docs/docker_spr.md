## Run the model

Download the backbone weights and set the `BACKBONE_WEIGHTS` environment variable
to point to the downloaded file:
```
curl -O https://download.pytorch.org/models/resnet34-333f7ec4.pth
export BACKBONE_WEIGHTS=$(pwd)/resnet34-333f7ec4.pth
```

After you've downloaded the backbone weights and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode>
using docker. Set environment variables to point to the COCO dataset directory,
weights, precision, and an output directory for logs. By default, the `run.sh`
script will run the `train_performance.sh` quickstart script. To run the `train_accuracy.sh`
script instead, specify that script name using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export DATASET_DIR=<path to the COCO dataset directory>
export BACKBONE_WEIGHTS=<path to the downloaded weights file>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with train_performance.sh quickstart script
./run.sh

# Run a different script by specifying the SCRIPT env var
SCRIPT=train_accuracy.sh ./run.sh
```
