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
#SBATCH --job-name=iaf2dl1
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=modelinfer_out_%j.txt
#SBATCH --error=modelinfer_err_%j.txt

hostname
root_parent=$1 # IO root
sample_name=$2 # sample name prefix
root_home=$root_parent
root_iodir=$root_home/experiments/debug
$root_parent/anaconda3/bin/conda init bash
source $root_parent/.bashrc
conda info -e
conda activate iaf2
cd $root_home/sources/af2pth
rm -f $root_home/logs/*
rm -f $root_iodir/timmers_*
bash multi_pytorch_jit_modelinfer.sh 4

while true; do
  if [ -f "$root_iodir/timmers_${sample_name}.txt" ]; then
    n=`cat $root_iodir/timmers_${sample_name}*|grep model_infer|wc -l`
  else
    n=0
  fi
  if [ $n -eq 4 ]; then
    break
  fi
  sleep 60
done
echo "iaf2_modelinfer_done"
