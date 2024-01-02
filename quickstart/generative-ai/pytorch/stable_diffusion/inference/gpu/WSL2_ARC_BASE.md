# Using Intel® Extension for PyTorch* Docker on Intel® Arc Series GPU 

## Overview

This document has instructions for running Intel® Extension for PyTorch Docker container on Intel® Arc Series GPU on Windows 11 with WSL2.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Arc Series A770 GPU |
| Operating System | Windows 11 with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) | 
| Drivers | [GPU drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html) |
| Software | [Docker Desktop](https://docs.docker.com/desktop/wsl) |

Here are additional pre-requisites to set ssh in order to access WSL2 remotely. 
After having WSL2 installed on Windows environment, with a default user account on WSL2 distro,

* On WSL, set up SSH Server:
```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install openssh-server -y
#set some port to be the SSH port for WSL:
# you may need to remove any default port from the config (usually Port 22) for example as follows,
echo "Port 2222" | sudo tee -a /etc/ssh/sshd_config
# start server
sudo service ssh start
```
* From Windows, open a port proxy:
```bash
# from administrator CMD:
# get IP of WSL2 instance
wsl hostname -I
# open port proxy
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=2222 connectaddress=<IP of WSL> connectport=2222
# add firewall rule for the proxy port:
netsh advfirewall firewall add rule name="Open Port 2222 for WSL2" dir=in action=allow protocol=TCP localport=2222
#You should now be able to connect directly to the WSL instance using the Windows host IP address and the proxy port:
ssh <Windows host IP>  -p 2222 -l <default WSL user>
```
**Notes:**
* The IP address for WSL changes each time it is restarted. The proxy port will need to be updated.

* In the case of multiple WSL distros installed, define the desired distro as the default before trying to connect
## Run Using Docker

### Set up Docker Image
```bash
docker pull intel/intel-extension-for-pytorch:2.1.10-xpu
```
### Run Docker Image

Use the following command to start Intel® Extension for PyTorch* GPU container. You can use -v option to mount your local directory into the container.

```bash
IMAGE_NAME=intel/intel-extension-for-pytorch:2.1.10-xpu
docker run \
        -it --rm \
        --device /dev/dxg \
        --volume /usr/lib/wsl:/usr/lib/wsl \
        IMAGE_NAME 
```
#### Verify if XPU is accessible from PyTorch:
You are inside the container now. Run the following command to verify XPU is visible to PyTorch:
```bash
python -c "import torch;print(torch.device('xpu'))"
```
Sample output looks like below:
```
xpu
```
Then, verify that the XPU device is available to Intel® Extension for PyTorch\*:
```bash
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"
```
Sample output looks like below:
```
True
```
Use the following command to check whether MKL is enabled as default:
```bash
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"
```
Sample output looks like below:
```
True
```
Finally, use the following command to show detailed information of detected device:
```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {ipex.xpu.get_device_properties(i)}') for i in range(ipex.xpu.device_count())];"
```
Sample output looks like below:
```bash
2.1.0a0+cxx11.abi
2.1.10+xpu
[0]: _DeviceProperties(name='Intel(R) Graphics [0x56a0]', platform_name='Intel(R) Level-Zero', dev_type='gpu, support_fp64=0, total_memory=13004MB, max_compute_units=512, gpu_eu_count=512)
```
