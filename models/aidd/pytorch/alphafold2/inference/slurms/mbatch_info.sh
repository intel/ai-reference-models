#!/bin/bash
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
#SBATCH --job-name=iaf2info
#SBATCH --partition=64c512g
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --output=mbatchinfo_out_%j.txt
#SBATCH --error=mbatchinfo_err_%j.txt


echo "----"
echo jobid=$SLURM_JOB_ID # job index
echo submitdir=$SLURM_SUBMIT_DIR # working directory of script that submit
echo job_nnodes=$SLURM_JOB_NUM_NODES # node number when task is allocated
echo job_nodelist=$SLURM_JOB_NODELIST # node list when task is allocated
echo job_cpupernode=$SLURM_JOB_CPUS_PER_NODE # allocated CPU cores in allocated task
echo "----"

nidsubstr=${SLURM_JOB_NODELIST:5:-1}
IFS=',' nidranges=(${nidsubstr})
nnodes=0
unset $nids
for nidrange in ${nidranges[@]}; do
  if [[ $nidrange == *"-"* ]]; then
    IFS='-' startend=(${nidrange})
    start=${startend[0]}
    end=${startend[1]}
    IFS=' ' subnodes=`seq $start $end|xargs`
    for nid in $subnodes; do
      nids[$nnodes]=$nid
      ((nnodes+=1))
    done
  else
    nids[$nnodes]=$nidrange
    ((nnodes+=1))
  fi
done

echo "total node number = $nnodes"
prefix="node"
for nid in ${nids[@]}; do
  echo allocate task on $prefix$nid
  srun -w $prefix$nid --exclusive bash slurms/print_node_info.sh &
done
