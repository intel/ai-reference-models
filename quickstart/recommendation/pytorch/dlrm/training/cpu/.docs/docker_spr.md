## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [downloaded dataset](#datasets), use the `run.sh` script from the container package
to run <model name> <mode>. Set environment variables to specify the dataset directory,
precision to run, and an output directory. The `run.sh` script will run the
`training.sh` quickstart script in the docker container. Set the number of
batches that are run using the `NUM_BATCH` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the quickstart script to test performance
NUM_BATCH=10000 ./run.sh

# Or, run quickstart script for testing convergence trend
NUM_BATCH=50000 ./run.sh
```
