<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/1.15.2/)
* numactl
* numpy==1.16.3
* Pillow>=9.3.0
* matplotlib
* click
* Clone the [tf_unet](https://github.com/jakeret/tf_unet) repository,
   and then get [PR #276](https://github.com/jakeret/tf_unet/pull/276)
   to get cpu optimizations:

   ```
   git clone https://github.com/jakeret/tf_unet.git
   cd tf_unet/
   git fetch origin pull/276/head:cpu_optimized
   git checkout cpu_optimized
   ``` 

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `TF_UNET_DIR` and an `OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
TF_UNET_DIR=<path to tf_unet directory>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/fp32_inference.sh
```
