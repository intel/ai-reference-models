# Using Intel® AI Reference Models in AI Tools

The Intel® AI Reference Models repository source code is bundled as part of the
[Intel® oneAPI AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html) (AI Tools).
Follow the instructions below to get your environment setup with AI Tools to run
TensorFlow models.

## Install AI Tools

Use the link below for instructions on how to install AI Tools or run the AI Tools
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

## Navigate to the AI Reference Models

Navigate to the Intel AI Reference Models source directory. It's located in your oneapi installation path, typically `/opt/intel/oneapi/ai_reference_models`.

Then browse to the `models` directory to run the model:
```
cd /opt/intel/oneapi/ai_reference_models/models
```

You can use this version of the AI Reference Models to run models instead of cloning the repository
from GitHub. The [README](/benchmarks/../README.md) has a list of the
available models and links to their instructions. Note that individual models
may require additional dependencies to be installed. You should follow the
bare metal installation instructions in those documents to run in the AI Tools
TensorFlow conda environment.
