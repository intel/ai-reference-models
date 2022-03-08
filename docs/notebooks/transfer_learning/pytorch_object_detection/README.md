# Transfer Learning for Object Detection using PyTorch

This notebook uses object detection models from torchvision that were originally trained 
using COCO and does transfer learning with the PennFudan dataset, consisting of 170 images 
with 345 labeled pedestrians.

The notebook performs the following steps:

1. Import dependencies and setup parameters
2. Prepare the dataset
3. Predict using the original model
4. Transfer learning
5. Visualize the model output
6. Export the saved model


## Running the notebook

The instructions below explain how to run the notebook on [bare metal](#bare-metal) using a
virtual environment.

1. Get a clone of the Model Zoo repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install `intel_extension_for_pytorch` and other required packages.
   
   You can use virtualenv:
   ```
   python3 -m venv intel-pyt-venv
   source intel-pyt-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt python=3.9
   conda activate intel-pyt
   ```
   Then, from inside the activated virtualenv or conda environment run these steps (note: if you are 
   inside the Intel firewall, you may need to remove intel.com from your no_proxy environment
   variable in order to apply proxying for the IPEX download):
   ```
   pip install --upgrade pip
   pip install intel_extension_for_pytorch==1.10.100 -f https://software.intel.com/ipex-whl-stable
   pip install -r ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_object_detection/requirements.txt
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the PennFudan 
   dataset to the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_object_detection
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.
   
Dataset Citation:

@InProceedings{10.1007/978-3-540-76386-4_17,
    author="Wang, Liming
    and Shi, Jianbo
    and Song, Gang
    and Shen, I-fan",
    editor="Yagi, Yasushi
    and Kang, Sing Bing
    and Kweon, In So
    and Zha, Hongbin",
    title="Object Detection Combining Recognition and Segmentation",
    booktitle="Computer Vision -- ACCV 2007",
    year="2007",
    publisher="Springer Berlin Heidelberg",
    address="Berlin, Heidelberg",
    pages="189--199",
    abstract="We develop an object detection method combining top-down recognition with bottom-up image segmentation. There are two main steps in this method: a hypothesis generation step and a verification step. In the top-down hypothesis generation step, we design an improved Shape Context feature, which is more robust to object deformation and background clutter. The improved Shape Context is used to generate a set of hypotheses of object locations and figure-ground masks, which have high recall and low precision rate. In the verification step, we first compute a set of feasible segmentations that are consistent with top-down object hypotheses, then we propose a False Positive Pruning(FPP) procedure to prune out false positives. We exploit the fact that false positive regions typically do not align with any feasible image segmentation. Experiments show that this simple framework is capable of achieving both high recall and high precision with only a few positive training examples and that this method can be generalized to many object classes.",
    isbn="978-3-540-76386-4"
}

