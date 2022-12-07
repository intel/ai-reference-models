<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install Intel Tensorflow
  ```bash
  pip install intel-tensorflow==2.11.dev202242
  ```

* Install the keras version that works with the above tensorflow version:
  ```bash
  pip install keras-nightly==2.11.0.dev2022092907
  ```

* Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: 
  ```bash
  DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  # To run VNNI, please set 
  DNNL_MAX_CPU_ISA=AVX512_CORE_BF16
  ```

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models)
  ```bash
  git clone https://github.com/IntelAI/models
  ```
