<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit on Linux</th>
    <th>Setup without AI Kit on Linux</th>
    <th>Setup without AI Kit on Windows</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit on Linux you will need:</p>
      <ul>
        <li>Cython
        <li>numactl
        <li>pandas
        <li>wget
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>Cython
        <li>git
        <li>numactl
        <li>pandas
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/tensorflow/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
        <li>Cython
        <li>pandas
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Once you've completed the setup, download and extract the pretrained model,
and set the file's path to the `PRETRAINED_MODEL` environment variable.
If you run on Windows, please use a browser to download the pretrained model using the link below. For Linux, run:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/transformer_lt_official_fp32_pretrained_model.tar.gz
tar -xzf transformer_lt_official_fp32_pretrained_model.tar.gz
export PRETRAINED_MODEL=$(pwd)/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb
```

Navigate to the model zoo directory, ensure that you have all the required
environment variables set, then run a [quickstart script](#quick-start-scripts) on either [Linux](#run-on-linux) or [Windows](#run-on-windows) systems.

### Run on Linux
```
# cd to your model zoo directory
cd models

# Set env vars
export DATASET_DIR=<path to the test dataset directory>
export PRETRAINED_MODEL=<path to the frozen graph .pb file>
export OUTPUT_DIR=<directory where the log and translation file will be written>

# Run a quickstart script
./quickstart/language_translation/tensorflow/transformer_lt_official/inference/cpu/fp32/<script name>.sh
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your model zoo directory
cd models

# Set env vars
set DATASET_DIR=<path to the test dataset directory>
set PRETRAINED_MODEL=<path to the frozen graph .pb file>
set OUTPUT_DIR=<directory where the log and translation file will be written>

# Run a quickstart script
bash quickstart\language_translation\tensorflow\transformer_lt_official\inference\cpu\fp32\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the pretrained model path on Windows is `D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\transformer_lt_official_fp32_pretrained_model\graph\fp32_graphdef.pb
> /d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb
>```
>Then, set the `PRETRAINED_MODEL` environment variable `set PRETRAINED_MODEL=/d/user/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb`.
