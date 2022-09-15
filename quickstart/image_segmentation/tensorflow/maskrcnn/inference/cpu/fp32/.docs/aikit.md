<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>AI Kit does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3.6 or 3.7
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>numactl
        <li>git
        <li>wget
        <li>IPython[all]
        <li>pillow>=8.1.2
        <li>cython
        <li>h5py==2.10.0
        <li>imgaug
        <li>keras==2.0.8
        <li>matplotlib
        <li>numpy==1.16.3
        <li>opencv-python
        <li>pycocotools
        <li>scikit-image
        <li>scipy==1.2.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Running <model name> also requires a clone and particular SHA of the
[Mask R-CNN model repository](https://github.com/matterport/Mask_RCNN). 
Set the `MODEL_SRC_DIR` env var to the path of your clone.
```
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
git checkout 3deaec5d902d16e1daf56b62d5971d428dc920bc
export MODEL_SRC_DIR=$(pwd)
```

Download pre-trained COCO weights `mask_rcnn_coco.h5)` from the
[Mask R-CNN repository release page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5),
and place it in the `MaskRCNN` directory.
```
wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
cd ..
```

After your environment is setup, set environment variables to the `DATASET_DIR`
and an `OUTPUT_DIR` where log files will be written. Ensure that you already have
the `MODEL_SRC_DIR` path set from the previous commands.
Once the environment variables are all set, you can run the
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<path to the directory where log files will be written>
export MODEL_SRC_DIR=<path to the Mask RCNN models repo with pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_segmentation/tensorflow/maskrcnn/inference/cpu/fp32/fp32_inference.sh
```
