<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
coco training dataset, TensorFlow models repo, and an output directory
where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export DATASET_DIR=<path to the training dataset directory>
export OUTPUT_DIR=<directory where log files will be written>
```

Use the command below to run <model name> <mode>. You can also edit parameters
like the `--num-train-steps` and `--mpi_num_processes` (to run multi-instance
training) as needed. If you want checkpoint files to be saved, specify the
`--checkpoint <directory path>` flag for the location where files will be written.

> Note: for best performance, use the same value for the arguments num-cores and num-intra-thread as follows:
>   For single instance run (mpi_num_processes=1): the value is equal to number of logical cores per socket.
>   For multi-instance run (mpi_num_processes > 1): the value is equal to (#_of_logical_cores_per_socket - 2).
>   If the `--num-cores` or `--num-intra-threads` args are not specified, these args will be calculated based on
>   the number of logical cores on your system.

```bash
python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --num-train-steps 100 \
    --num-cores 52 \
    --num-inter-threads 1 \
    --num-intra-threads 52 \
    --batch-size=100 \
    --weight_decay=1e-4 \
    --mpi_num_processes=1 \
    --mpi_num_processes_per_socket=1 \
    --output-dir ${OUTPUT_DIR} \
    --docker-image <docker image>
```
