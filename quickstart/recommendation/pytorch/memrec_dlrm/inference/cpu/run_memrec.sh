export MODEL_DIR="frameworks.ai.models.intel-models"

# Env vars
export PRECISION="fp32"
export WEIGHT_PATH="memrec_CriteoTB_D75000dw75000_K1kw1.pt" #only needed for testing accuracy
export DATASET_DIR="data/TB/preprocessed"
export OUTPUT_DIR="memrec_out"

# Run a quickstart script (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/memrec_dlrm/inference/cpu

#bash inference_performance.sh
bash accuracy.sh
