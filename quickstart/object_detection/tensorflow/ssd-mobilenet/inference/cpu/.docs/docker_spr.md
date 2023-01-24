## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run <model name> <mode> in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required only for the accuracy script.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

SCRIPT=accuracy.sh ./run.sh
```