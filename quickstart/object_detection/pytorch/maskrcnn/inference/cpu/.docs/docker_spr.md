## Run the model

Download the pretrained model and set the `CHECKPOINT_DIR` environment variable
to point to the directory where the weights file is downloaded.
```
curl -O https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
export CHECKPOINT_DIR=$(pwd)
```

After you've downloaded the pretrained model and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode>.
Set environment variables to specify the dataset directory, precision and mode to run, and
an output directory. By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export MODE=<imperative or jit>
export CHECKPOINT_DIR=<path to the directory where the downloaded .pth file is located>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Specify a different quickstart script to run, for example, accuracy.sh
SCRIPT=accuracy.sh ./run.sh
```
