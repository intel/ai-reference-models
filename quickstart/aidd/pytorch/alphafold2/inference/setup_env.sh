# Copyright (c) 2022 Intel Corporation
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

### hyperparams zone start ###
root_home=$1 # e.g. /mnt/data1/demohome, path that holds all input/intermediates/output data
refdata_dir=$2 # e.g. /mnt/data1, path of alphafold reference dataset, including bfd, uniref, params etc.
conda_env_name=$3 # e.g. iaf2, alphafold2 conda env
experiment_name=$4 # e.g. debug, project name (it may include multiple different samples)
model_name=$5 # e.g. model_1, alphafold weight prefix selected for model inference
### hyperparams zone end   ###

# check if essential hyperparams are set
if [[ $root_home == "null" ]]; then
  echo "### <ERROR> please provide a path that holds all input/intermediates/output data. exiting"
  exit
elif [[ $refdata_dir == "null" ]]; then
  echo "### <ERROR> please provide a path of alphafold reference dataset, including bfd, uniref, params etc.. exiting"
  exit
fi

# check and create directories for I/O data
weights_dir=$root_home/weights
extracted_weights_dir=$weights_dir/extracted
log_dir=$root_home/logs
samples_dir=$root_home/samples
result_dir=$root_home/experiments
experiment_dir=$root_home/experiments/$experiment_name

if [ -d $root_home ]; then # validate home path
  if [ ! -d $samples_dir ]; then
    mkdir $samples_dir
  fi
  # enumerate and create every subfolder 
  for d in $weights_dir $extracted_weights_dir $log_dir $result_dir $experiment_dir; do
    if [ ! -d $d ]; then
      mkdir $d
    fi
  done
else # no access to home path
  echo "### <ERROR> invalid root_home directory: $root_home. exiting"
  exit
fi

# validate all reference data of alphafold2
ref_bfd_dir=$refdata_dir/bfd
ref_mgnify_dir=$refdata_dir/mgnify
ref_pdb70_dir=$refdata_dir/pdb70
ref_pdbmmcif_dir=$refdata_dir/pdb_mmcif
ref_uniclust30_dir=$refdata_dir/uniclust30
ref_uniref90_dir=$refdata_dir/uniref90
ref_params_dir=$refdata_dir/params
if [ -d $refdata_dir ]; then # validate reference dataset of alphafold2
  for d in $ref_bfd_dir $ref_mgnify_dir $ref_pdbmmcif_dir $ref_pdb70_dir $ref_uniclust30_dir $ref_uniref90_dir $ref_params_dir; do
    if [ ! -d $d ]; then
      echo " ## <ERROR> invalid reference data folder $d. exiting"
      exit
    fi
  done
  # validate bfd
  n_ffdata=`ls $ref_bfd_dir|grep "ffdata$"|wc -l`
  n_ffidx=`ls $ref_bfd_dir|grep "ffindex$"|wc -l`
  if [[ $n_ffdata != 3 ]]; then
    echo " ## <ERROR> incomplete ffdata in BFD dataset, please check folder $ref_bfd_dir"
    exit
  fi
  if [[ $n_ffidx != 3 ]]; then
    echo " ## <ERROR> incomplete ffindex in BFD dataset, please check folder $ref_bfd_dir. exiting"
    exit
  fi
  # validate mgnify
  n_fa=`ls $ref_mgnify_dir|grep "fa$"|wc -l`
  n_fasta=`ls $ref_mgnify_dir|grep "fasta$"|wc -l`
  if [ 0 -lt $n_fa ]; then
    f_mgnify=$ref_mgnify_dir/`ls $ref_mgnify_dir|grep "fa$"`
  elif [ 0 -lt $n_fasta ]; then
    f_mgnify=$ref_mgnify_dir/`ls $ref_mgnify_dir|grep "fasta$"`
  else
    echo " ## <ERROR> invalid mgnify dataset, please check folder $ref_mgnify_dir. exiting"
    exit
  fi
  # validate pdb70
  n_pdb70_files=`ls $ref_pdb70_dir|wc -l`
  if [ $n_pdb70_files -lt 9 ]; then
    echo " ## <ERROR> incomplete pdb70 dataset, please check folder $ref_pdb70_dir. exiting"
    exit
  fi
  # validate pdb_mmcif
  f_obs="$ref_pdbmmcif_dir/obsolete.dat"
  mmcif_dir="$ref_pdbmmcif_dir/mmcif_files"
  n_mmcif=`ls $mmcif_dir|wc -l`
  if [ $n_mmcif -lt 180000 ]; then
    echo " ## <ERROR> incomplete pdb mmcif dataset, please check folder $mmcif_dir. exiting"
    exit
  elif [ ! -f $f_obs ]; then
    echo " ## <ERROR> missing $f_obs, please check folder $ref_pdbmmcif_dir, exiting"
    exit
  fi
  # validate uniclust30
  uniclust30_dir=$ref_uniclust30_dir/`ls $ref_uniclust30_dir`
  n_uniclust30_files=`ls $uniclust30_dir|wc -l`
  if  [ $n_uniclust30_files -lt 13 ]; then
    echo " ## <ERROR> incomplete uniclust30 dataset, please check folder $uniclust30_dir. exiting"
    exit
  fi
  # validate uniref90
  f_uniref90="$ref_uniref90_dir/uniref90.fasta"
  if [ ! -f $f_uniref90 ]; then
    echo " ## <ERROR> invalid uniref90 dataset, please check folder $ref_uniref90_dir. exiting"
    exit
  fi
  # validate model params
  f_params="${ref_params_dir}/params_${model_name}.npz"
  if [ ! -f $f_params ]; then
    echo " ## <ERROR> invalid params folder $ref_params_dir or invalid model name $model_name. exiting"
    exit
  fi
fi

# check conda & python
if [[ `which conda` != *"conda" ]]; then
  echo "### <ERROR> conda cmd not found. exiting"
  exit
else
  if [[ `python --version` != *"Intel"* ]]; then
    echo " ## <INFO> installing intel python env"
    conda install -y -c intel intelpython
    conda install -y -c intel python
  fi
fi

# create conda env
echo " ## <INFO> installing alphafold2 conda env"
conda install -y -c conda-forge openmm pdbfixer aria2
conda install -y -c conda-forge -c bioconda hhsuite
conda install -y -c bioconda hmmer kalign2
conda install -y -c pytorch pytorch cpuonly
conda install -y jemalloc

# download source code
export IAF2_HOME=`pwd`

# install pip dependencies
python -m pip install absl-py biopython chex dm-haiku dm-tree immutabledict jax ml-collections numpy scipy tensorflow pandas psutil tqdm joblib
python -m pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install ipex >= 1.11.100
if [[ `python -m pip list|grep intel` == *"intel-extension-for-pytorch"* ]]; then
  has_ipex=1
else
  echo "## <INFO> IPEX not exists, will install latest compatible ipex"
  has_ipex=0
  python -m pip install intel_extension_for_pytorch
  if [[ `python -m pip list|grep intel` != *"intel-extension-for-pytorch"* ]]; then
    has_ipex=1
  fi
fi

# extract model
echo "### <INFO> extracting model parameter file"
if [ ! -d $weights_dir ]; then
  mkdir $weights_dir
fi
if [ ! -d $extracted_weights_dir ]; then
  mkdir ${extracted_weights_dir}/${model_name}
fi
if [ ! -f "extract_params.py" ]; then
  cd $(dirname $0)/../../../../../models/aidd/pytorch/alphafold2/inference
fi
git submodule update --init --recursive
cd alphafold
git checkout v2.0.1
cd -
if [ ! -d "alphafold/data" ]; then
  echo "### <ERROR> submodule initiation of DeepMind AlphaFold2 repo failed."
  echo " ## <INFO> Please clone DeepMind repo as subfolder 'alphafold' under install root. exiting."
  exit
fi
python extract_params.py --input $f_params --output_dir ${extracted_weights_dir}/${model_name}
cd $IAF2_HOME

echo "### <INFO> Initialization of alphafold2 is done."
echo " ## <WARNING> Please ensure there is at least one input sample file (*.fa) under $samples_dir!"
echo "### <INFO> We put a sample.fa to $samples_dir to help you understand how it works."
curl https://rest.uniprot.org/uniprotkb/Q6UWK7.fasta > sample.fa
cp sample.fa $samples_dir/
