set -e

echo "Setup PyTorch Test Enviroment for RNNT Training"

PRECISION=$1
OUTPUT_DIR=${OUTPUT_DIR-"$(pwd)/tests/cicd/pytorch/rnnt/training/cpu/output/${PRECISION}"}
is_lkg_drop=$2
DATASET_DIR=$3
DISTRIBUTED=$4
profiling=$5

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${is_lkg_drop}" == "true" ]]; then
  source ${WORKSPACE}/pytorch_setup/bin/activate pytorch
fi

export LD_PRELOAD="${WORKSPACE}/jemalloc/lib/libjemalloc.so":"${WORKSPACE}/tcmalloc/lib/libtcmalloc.so":"/usr/local/lib/libiomp5.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# Install dependency
cd models_v2/pytorch/rnnt/training/cpu
MODEL_DIR=${MODEL_DIR}
./setup.sh

OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET_DIR} DISTRIBUTED=${DISTRIBUTED} MODEL_DIR=${MODEL_DIR} profiling=${profiling} ./run_model.sh
cd -
