## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [downloaded dataset](#datasets), use the `run.sh` script from the container package
to run <model name> <mode>. Set environment variables to specify the dataset directory,
precision to run, and an output directory. By default, the `run.sh` script will run the
`training_performance.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable. You can optionally change
the number of batches that are run using the `NUM_BATCH` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training_performance.sh quickstart script
./run.sh

# Specify a different quickstart script to run
SCRIPT=training_convergence.sh ./run.sh

# Specify a custom number of batches
NUM_BATCH=20000 ./run.sh
```
