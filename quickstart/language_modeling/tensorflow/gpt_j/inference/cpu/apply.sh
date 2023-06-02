WORKING_DIRECTORY=./quickstart/language_modeling/tensorflow/gpt_j/inference/cpu/

cd ${WORKING_DIRECTORY}
rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cp ./tf_transformers_metrics.patch ./transformers/
cd transformers
git checkout v4.28-release
git apply tf_transformers_metrics.patch
pip install -e .