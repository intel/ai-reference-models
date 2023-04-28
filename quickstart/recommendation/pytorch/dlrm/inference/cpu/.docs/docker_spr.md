## Run the model

Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the
[MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights)
and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the tb00_40M.pt file>
```

After you've downloaded the model weight file and followed the instructions to
[build the container](#build-the-container) and [downloaded dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode>.
Set environment variables to specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`inference_performance.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export WEIGHT_PATH=<path to the tb00_40M.pt file>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_performance.sh quickstart script
./run.sh

# Specify a different quickstart script to run, for example, accuracy.sh
SCRIPT=accuracy.sh ./run.sh
```
