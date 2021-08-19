## Run the model

Download the pretrained model and set the `PRETRAINED_MODEL` environment variable
to point to the file.
```
curl -O https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
export PRETRAINED_MODEL=$(pwd)/e2e_mask_rcnn_R_50_FPN_1x.pth
```

After you've downloaded the pretrained model and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode>.
Set environment variables to specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export PRETRAINED_MODEL=<path to the downloaded .pth file>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Specify a different quickstart script to run, for example, accuracy.sh
SCRIPT=accuracy.sh ./run.sh
```
