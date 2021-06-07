<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* librosa==0.5
* numactl
* Clone the [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
repo and get pull request #352 for the CPU optimizations.  The path to
the cloned repo will be passed as the model source directory when
running the launch script.

```
git clone https://github.com/ibab/tensorflow-wavenet.git
cd tensorflow-wavenet/

git fetch origin pull/352/head:cpu_optimized
git checkout cpu_optimized
```

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `TF_WAVENET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
export TF_WAVENET_DIR=<tensorflow-wavenet directory>
export OUTPUT_DIR=<directory where log files will be written>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/fp32_inference.sh
```
