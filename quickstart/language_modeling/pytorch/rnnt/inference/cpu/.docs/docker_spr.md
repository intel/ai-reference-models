## Run the model

Download the pretrained model and set the `PRETRAINED_MODEL` environment variable
to point to the file:
```
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O rnnt.pt
export PRETRAINED_MODEL=$(pwd)/rnnt.pt
```

After you've downloaded the pretrained model and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run <model name> <mode> in
docker. Set environment variables to specify the dataset directory, precision to run,
and an output directory. By default, the `inference_realtime.sh` quickstart script will
be run. To run a different quickstart script, set the `SCRIPT` environment variable
to the script of your choice.

The snippet below demonstrates how to run <model name> <mode>:
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRETRAINED_MODEL=<path to the rnnt.pt file>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Run a different quickstart script
SCRIPT=accuracy.sh ./run.sh
```
