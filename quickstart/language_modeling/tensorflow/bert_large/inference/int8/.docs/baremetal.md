<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* unzip

Once the above dependencies have been installed, download and untar the model
package, set environment variables, and then run a quickstart script. See the
[datasets](#datasets) and [list of quickstart scripts](#quick-start-scripts) 
for more details on the different options.

The snippet below shows how to run a quickstart script:
```
wget <package url>
tar -xvf <package name>
cd <package dir>

DATASET_DIR=<path to the SQuAD dataset>
OUTPUT_DIR=<directory where log files will be saved>

# Run a script for your desired usage
bash ./quickstart/<script name>.sh
```
