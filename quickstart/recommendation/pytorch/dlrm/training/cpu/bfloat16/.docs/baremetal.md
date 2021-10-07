<!--- 40. Bare Metal -->
## Bare Metal

To run on bare metal first, follow the [instruction described here](/models/recommendation/pytorch/dlrm/training/bfloat16/README.md#1-install-anaconda-30) until section 4.

After installing the prerequisites, download and untar the DLRM bf16 training model package, and run the quick start script:
```
wget <package url>
tar -xvf <package name>
```
Set environment variables
for the path to your `DATA_PATH`then run a 
[quickstart script](#quick-start-scripts).

```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

cd <package directory>
bash quickstart/<script name>.sh
```
