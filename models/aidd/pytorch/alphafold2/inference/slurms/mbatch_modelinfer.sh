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
#SBATCH --job-name=diaf2dl
#SBATCH --partition=64c512g
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH --output=mbatchinfo_out_%j.txt
#SBATCH --error=mbatchinfo_err_%j.txt

echo "### model inference on multi nodes"
echo "----"
echo jobid=$SLURM_JOB_ID # job id
echo submitdir=$SLURM_SUBMIT_DIR # task pwd
echo job_nnodes=$SLURM_JOB_NUM_NODES # #nodes allocated
echo job_nodelist=$SLURM_JOB_NODELIST # list off nodes allocated
echo job_cpupernode=$SLURM_JOB_CPUS_PER_NODE # CPU core/node
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
procid=0
for nid in ${nids[@]}; do
  if [ $nid -lt 10 ]; then
    nid="00$nid"
  elif [ $nid -lt 100 ]; then
    nid="0$nid"
  else
    nid="$nid"
  fi
  ((start=$procid*4))
  echo allocate task on $prefix$nid start_idx is $start
  srun -p 64c512g -w $prefix$nid bash multi_pytorch_jit_modelinfer.sh $start 4 &
  ((procid+=1))
done

while true; do
  sleep 60
  f_log="$SLURM_SUBMIT_DIR/mbatchinfo_out_$SLURM_JOB_ID.txt"
  n=0
  if [ -f $f_log ]; then
    n=`cat $f_log|grep iaf2_modelinfer_done|wc -l`
  else
    echo "[warning] no valid slurm log: $f_log"
  fi
  if [ $n -eq $SLURM_JOB_NUM_NODES ]; then
    echo "all nodes completed"
    break
  fi
done
