WORKING_DIRECTORY=./quickstart/language_modeling/tensorflow/gpt_j/inference/cpu/

cd ${WORKING_DIRECTORY}
rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cp ./tf_v30_metrics.patch ./transformers/
cd transformers
git checkout v4.30-release
git apply tf_v30_metrics.patch
pip install -e .