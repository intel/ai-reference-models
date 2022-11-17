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
#SBATCH --job-name=iaf2msa1
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=preproc_onenode_out.txt
#SBATCH --error=preproc_onenode_err.txt

root_home=$1 # root of IO paths
sample_name=$2 # sample prefix
hostname
rm -f $root_home/logs/*
source $root_home/.bashrc
cd $root_home/af2pth
conda activate iaf2
bash multi_preproc.sh 8

while true; do
  if [ -f "$root_home/logs/${sample_name}.fa.txt" ]; then
    n=`cat $root_home/${sample_name}*|grep iaf2_preproc_done|wc -l`
  else
    n=0
  fi
  if [ $n -eq 8 ]; then
    break
  fi
  sleep 60
done
echo "iaf2_preproc_done"
