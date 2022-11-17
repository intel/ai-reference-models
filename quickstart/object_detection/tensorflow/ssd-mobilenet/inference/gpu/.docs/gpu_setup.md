<!--- 20. GPU Setup -->
## Hardware Requirements:
- Intel® Data Center GPU Flex Series

## Software Requirements:
- Ubuntu 20.04 (64-bit)
- Intel GPU Drivers: Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)

  |Release|OS|Intel GPU|Install Intel GPU Driver|
    |-|-|-|-|
    |v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

- Intel® oneAPI Base Toolkit 2022.3: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Math Kernel Library (oneMKL)
  * Download and install the verified DPC++ compiler and oneMKL in Ubuntu 20.04.

    ```bash
    $ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
    # 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, Threading Building Blocks and oneMKL
    $ sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
    ```
    For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

  - Set environment variables
    Default installation location {ONEAPI_ROOT} is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts
    ```bash
    source {ONEAPI_ROOT}/setvars.sh
    ```
