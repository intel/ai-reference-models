## Enviromnment setup

* Create a virtual environment `venv-tf` using `Python 3.8`:
```
pip install virtualenv
# use `whereis python` to find the `python3.8` path in the system and specify it. Please install `Python3.8` if not installed on your system.
virtualenv -p /usr/bin/python3.8 venv-tf
source venv-tf/bin/activate
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow/2.11.dev202242/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow==2.11.dev202242
pip install keras-nightly==2.11.0.dev2022092907
```
> Note: For `kernel version 5.16`, `AVX512_CORE_AMX` is turned on by default. If the `kernel version < 5.16` , please set the following environment variable for AMX environment: `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. To run VNNI, please set `DNNL_MAX_CPU_ISA=AVX512_CORE_BF16`.

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.
