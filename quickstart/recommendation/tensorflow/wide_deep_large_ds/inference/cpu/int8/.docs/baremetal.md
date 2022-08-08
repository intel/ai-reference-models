<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>
```

* Running inference to check accuracy:
```
./quickstart/int8_accuracy.sh
```

* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.
```
NUM_OMP_THREADS=1
./quickstart/int8_online_inference.sh \
--num-intra-threads 1 --num-inter-threads 1
```

