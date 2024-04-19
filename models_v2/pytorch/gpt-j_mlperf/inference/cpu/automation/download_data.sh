#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e
set +x

: ${model=${1:-""}}
: ${dtype=${2:-"int8"}}
: ${output_dir=${3:-"/data/mlperf_data1"}}
: ${conda_path=${4:-"${HOME}/miniconda3"}}

echo "model: ${model}, dtype: ${dtype}, output_dir: ${output_dir}, conda_path:${conda_path}"

SUPPORTED_MODELS=("resnet50" "retinanet" "rnnt" "3d-unet" "bert" "gpt-j" "dlrm_2" "stable_diffusion" "all")
WORK_DIR=`pwd`



function check_model() {
    if [[ "${model}" == "" ]] || [[ ! " ${SUPPORTED_MODELS[@]} " =~ " ${model} " ]]; then
        read -p "which model?(${SUPPORTED_MODELS[*]}):" model
        check_model ${model}
    fi    
}

function ask_clean_all() {
    local c=$1
    local choises=("y" "n")
    if [[ " ${choises[@]} " =~ " ${c} " ]]; then
        if [ "$c" == "y" ]; then
            rm -rf ${output_dir}
        else
            echo "canceled."
        fi
    else
        read -p "y or n:" c1
        ask_clean_all $c1
    fi
}

function clean() {
    if [ "${model}" == "all" ]; then
        read -p "Do you want to delete all data? [y/n]" choise
        ask_clean_all $choise
    else
        if [ "${model}" == "3d-unet" ]; then
            local data_dir=${output_dir}/3dunet-kits
        else
            local data_dir=${output_dir}/${model}
        fi
        if [ -d ${data_dir} ]; then
            rm -rf ${data_dir} 
        fi
        mkdir -p ${data_dir}
    fi
}

function download_data() {
    local arg url datadir output_fname decompress_codec decompress_dir
    url=$1
    datadir=$2
    output_fname=$3
    decompress_codec=$4
    decompress_dir=$5

    pushd ${datadir}
    if [ -z ${output_fname} ]; then
        echo "Err: output file name not found."
        exit 1
    else
        if [ -f ${output_fname} ]; then
            echo "Skip downloading because the file exists: ${output_fname}"
        else
            wget --no-check-certificate ${url} -O ${output_fname}
        fi
    fi

    if [ ! -z ${decompress_codec} ]; then
        if [ -d ${decompress_dir} ]; then
            rm -rf ${decompress_dir}
        fi
        mkdir -p ${decompress_dir}
        echo "Decompressing..."
        case ${decompress_codec} in 
            "tar")
                tar -xvf ${output_fname} -C ${decompress_dir}
                ;;
            "zip")
                unzip ${output_fname} -d ${decompress_dir}
        esac
        rm -f ${output_fname}
    fi
    popd
}

function download_resnet50() {
    local data_dir=${output_dir}/resnet50
    RESNET50_CODE_DIR="${WORK_DIR}/../resnet50/pytorch-cpu"
    mkdir -p ${data_dir}
    echo "Downloading dataset ILSVRC2012..."
    download_data "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" \
                  ${data_dir} \
                  "ILSVRC2012_img_val.tar" \
                  "tar" \
                  "${data_dir}/ILSVRC2012_img_val"
    cp ${RESNET50_CODE_DIR}/prepare_calibration_dataset.sh ${data_dir}/
    cd ${data_dir}/
    bash prepare_calibration_dataset.sh

    echo "Downloading resnet50 FP32 model..."
    download_data "https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth" \
                  ${data_dir} \
                  "${data_dir}/resnet50-fp32-model.pth"
}

function download_retinanet() {
    local data_dir=${output_dir}/retinanet/data
    mkdir -p ${data_dir}

    echo "Downloading OpenImages (264)..."
    source ${conda_path}/etc/profile.d/conda.sh
    conda env remove -n mlperf_retinanet_proc
    conda create -n mlperf_retinanet_proc python=3.9 --yes
    conda activate mlperf_retinanet_proc
    conda info
    python -m pip install fiftyone==0.17.2

    pushd ${WORK_DIR}/../retinanet/pytorch-cpu
    bash openimages_mlperf.sh --dataset-path ${data_dir}/openimages

    echo "Downloading Calibration images..."
    bash openimages_calibration_mlperf.sh --dataset-path ${data_dir}/openimages-calibration

    conda deactivate
    popd

    echo "Downloading retinanet model..."
    download_data "https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth" \
                  ${data_dir} \
                  "${data_dir}/retinanet-model.pth"
}

function download_rnnt() {
    local data_dir=${output_dir}/rnnt/mlperf-rnnt-librispeech
    mkdir -p ${data_dir}

    echo "Downloading rnnt dataset..."
    source ${conda_path}/etc/profile.d/conda.sh
    conda env remove -n mlperf_rnnt_proc
    conda create -n mlperf_rnnt_proc python=3.9 --yes
    conda activate mlperf_rnnt_proc
    conda info
    python -m pip install pandas requests tqdm toml

    pushd ${WORK_DIR}/../rnnt/pytorch-cpu
    LOCAL_DATA_DIR=${data_dir}/local_data
    mkdir -p ${LOCAL_DATA_DIR}/LibriSpeech ${LOCAL_DATA_DIR}/raw
    python datasets/download_librispeech.py \
            --input_csv=configs/librispeech-inference.csv \
            --download_dir=${LOCAL_DATA_DIR}/LibriSpeech \
            --extract_dir=${LOCAL_DATA_DIR}/raw
    conda deactivate
    popd

    echo "Downloading rnnt model..."
    download_data "https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1" \
                  ${data_dir} \
                  "${data_dir}/rnnt.pt"
}

function download_3dunet() {
    local data_dir=${output_dir}/3dunet-kits
    mkdir -p ${data_dir}

    echo "Downloading 3dunet dataset: KiTS19..."
    source ${conda_path}/etc/profile.d/conda.sh
    conda env remove -n mlperf_3dunet_proc
    conda create -n mlperf_3dunet_proc python=3.9 --yes
    conda activate mlperf_3dunet_proc
    conda info

    cd ${data_dir}
    git clone https://github.com/neheller/kits19
    pushd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    conda deactivate
    popd
}

function download_bert() {
    local data_dir=${output_dir}/bert
    local dataset_dir=${data_dir}/dataset
    local model_dir=${data_dir}/model
    mkdir -p ${dataset_dir}

    echo "Downloading bert dataset..."
    download_data "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" \
                  "${dataset_dir}" \
                  "${dataset_dir}/dev-v1.1.json"

    echo "Downloading bert model..."
    git clone https://huggingface.co/bert-large-uncased ${model_dir}
    rm -f ${model_dir}/pytorch_model.bin
    download_data "https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1" \
                  "${model_dir}" \
                  "${model_dir}/pytorch_model.bin"
}

function download_dlrm2() {
    local data_dir=${output_dir}/dlrm_2
    local dataset_dir=${data_dir}/data_npy
    local model_dir=${data_dir}/model
    mkdir -p ${model_dir}

    echo "Downloading dlrm2 model..."
    pushd ${model_dir}
    download_data "https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download" \
                  ${model_dir} \
                  "${model_dir}/weigths.zip" \
                  "zip" \
                  "${model_dir}"
    
    echo "!!! You need to download dlrm2 dataset by yourself by using the following url, and decompress it to ${dataset_dir}. !!!"
    echo "https://labs.criteo.com/2013/12/download-terabyte-click-logs"
    echo "Move the decompressed files, finally you should see the following files in ${dataset_dir}:"
    echo "day_23_dense.npy"
    echo "day_23_labels.npy"
    echo "day_23_sparse_multi_hot.npy"
    echo "day_23_sparse.npy"
    read -p "Did you download the dataset? [y/n]" answer
    if [ "${answer}" == 'y' ];then
        echo "Great, let's continue."
    else
        echo "You must download it first."
        exit 1
    fi
}

function download_gptj() {
    local data_dir=${output_dir}/gpt-j
    local dataset_dir=${data_dir}/data
    local model_dir=${data_dir}/data/gpt-j-checkpoint
    GPTJ_CODE_DIR="${WORK_DIR}/../gptj-99/pytorch-cpu"
    GPTJ_CALIBRATION_DIR="${WORK_DIR}/../../calibration/gpt-j/pytorch-cpu"
    mkdir -p ${model_dir}

    echo "Downloading gpt-j model..."
    download_data "https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download" \
                  ${data_dir} \
                  "${data_dir}/gpt-j-checkpoint.zip" \
                  "zip" \
                  "${model_dir}"

    mv ${model_dir}/gpt-j/checkpoint-final/* ${model_dir}

    source ${conda_path}/etc/profile.d/conda.sh
    conda env remove -n mlperf_gpt-j_int8_proc
    conda create -n mlperf_gpt-j_int8_proc python=3.9 --yes
    conda activate mlperf_gpt-j_int8_proc
    conda info
    python -m pip install tqdm datasets

    echo "Downloading datasets..."
    pushd ${GPTJ_CODE_DIR}
    python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${dataset_dir}/calibration-data
    python download-dataset.py --split validation --output-dir ${dataset_dir}/validation-data
    conda deactivate
    popd

}

function download_stable_diffusion() {
    local data_dir=${output_dir}/stable_diffusion
    local dataset_dir=${data_dir}/data
    local model_dir=${data_dir}/model
    SD_CODE_DIR="${WORK_DIR}/../stable_diffusion/pytorch-cpu"
    mkdir -p ${model_dir}

    echo "ERROR!  Fail to Download stable_diffusion model..."

}

check_model

if [ ! -d ${conda_path} ]; then
    echo "Conda path not found: ${conda_path}. Use the following command to sepecify your conda path:"
    echo "conda_path=<your conda path> bash download_data.sh"
fi

# clean

echo "Downloading data and model for ${model}..."
case $model in
 
    "resnet50")
        download_resnet50
        ;;
    "retinanet")
        download_retinanet
        ;;
    "rnnt")
        download_rnnt
        ;;
    "3d-unet")
        download_3dunet
        ;;
    "bert")
        download_bert
        ;;
    "dlrm_2")
        download_dlrm2
        ;;
    "gpt-j")
        download_gptj
        ;;
    "stable_diffusion")
        download_stable_diffusion
        ;;
    "all")
        download_resnet50
        download_retinanet
        download_rnnt
        download_3dunet
        download_bert
        download_dlrm2
        download_gptj
        download_stable_diffusion
        ;;
    *)
        echo "No such model implementation, skipped."
        ;;
esac
echo "Done."
