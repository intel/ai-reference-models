## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run <model name> <mode> in docker. Set environment variables to
specify the dataset directory, precision to run, and an output directory.

```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training.sh quickstart script
./run.sh
```
