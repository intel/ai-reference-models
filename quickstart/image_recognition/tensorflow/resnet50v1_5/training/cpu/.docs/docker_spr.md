## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run <model name> <mode> in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`multi_instance_training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-resnet50v1-5-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the container with training.sh quickstart script
./run.sh
```
