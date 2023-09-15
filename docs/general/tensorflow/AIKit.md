# Using the Model Zoo in the Intel® oneAPI AI Analytics Toolkit

The Model Zoo is bundled as part of the
[Intel® oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).
Follow the instructions below to get your environment setup with AI Kit to run
TensorFlow models.

## Install AI Kit

Use the link below for instructions on how to install AI Kit or run the AI Kit
docker container:
[https://www.intel.com/content/www/us/en/docs/oneapi-ai-analytics-toolkit/get-started-guide-linux/2023-1/overview.html](https://www.intel.com/content/www/us/en/docs/oneapi-ai-analytics-toolkit/get-started-guide-linux/2023-1/overview.html)

## Activate a Conda Environment

Use the `conda env list` command to see a list of all of the available conda environments.
The individual [model documentation files](/benchmarks#tensorflow-use-cases) will have
information on which conda environment can be used to run that particular model. There are
conda enviornments TensorFlow as well as for each deep learning use case (like image
recognition, recommendation, language translation, etc).

To activate a general use TensorFlow conda environment use the instructions below:

* **Activate conda environment With Root Access**
  Navigate the Linux shell to your oneapi installation path, typically `/opt/intel/oneapi`.
  Activate the conda environment with the following command:
  ```
  conda activate tensorflow
  ```
* **Activate conda environment Without Root Access (Optional)**
  By default, the Intel AI Analytics toolkit is installed in the `/opt/intel/oneapi` folder,
  which requires root privileges to manage it. If you would like to bypass using root access
  to manage your conda environment, then you can clone and activate the conda environment using
  the following commands:
  ```
  conda create --name user_tensorflow --clone tensorflow

  conda activate user_tensorflow
  ```

As a sanity check, use the following command to verify that TensorFlow can be imported and
oneDNN optimizations are enabled (this should print `oneDNN optimizations enabled: True`).
```
python -c "from tensorflow.python.util import _pywrap_util_port; print('oneDNN optimizations enabled:', _pywrap_util_port.IsMklEnabled())"
```

## Navigate to the Model Zoo

Navigate to the Intel Model Zoo source directory. It's located in your oneapi installation path,
typically `/opt/intel/oneapi/modelzoo`. You can view the available Model Zoo release versions
for the Intel AI Analytics toolkit:

```
ls /opt/intel/oneapi/modelzoo
2.4.0  latest
```

Then browse to the `models` directory in your preferred Intel Model Zoo release version location
to run the model:
```
cd /opt/intel/oneapi/modelzoo/latest/models
```

You can use this version of the zoo to run models instead of cloning the repository
from git. The [benchmarks README](/benchmarks/README.md) has a list of the
available models and links to their instructions. Note that individual models
may require additional dependencies to be installed. You should follow the
bare metal installation instructions in those documents to run in the AI Kit
TF conda environment.
