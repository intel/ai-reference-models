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

### input env params
root_condaenv=$1 # e.g. /home/<your-username>/anaconda3/envs/<your-envname>, root path of anaconda environment
root_home=$2 # e.g. /home/yangw, root path that holds all intermediate IO data
root_data=$3 # e.g. $root_home/af2data, path that holds all reference database and model params, including mgnify uniref etc.
input_dir=$4 # e.g. $root_home/samples, path of all query .fa files (sequences in fasta format)
out_dir=$5 # e.g. $root_home/experiments, path that contains intermediates output of preprocessing, model inference, and final result
model_name=$6 # e.g. model_1, the chosen model name of Alphafold2

data_dir=$root_data
log_dir=$root_home/logs
suffix=".fa"
n_sample=`ls ${input_dir}|grep ${suffix}|wc -l`
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((core_per_instance=$n_core*$n_socket/$n_sample))
((n_sample_0=$n_sample-1))
script="python run_modelinfer_pytorch_jit.py"
root_params=$root_home/weights/extracted/${model_name}
workdir=`pwd`
if [ ! -f "run_modelinfer_pytorch_jit.py" ]; then
  cd $(dirname $0)/../../../../../models/aidd/pytorch/alphafold2/inference
fi

export LD_PRELOAD=$root_condaenv/lib/libiomp5.so:$root_condaenv/lib/libjemalloc.so:$LD_PRELOAD
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export OMP_NUM_THREADS=$core_per_instance
export TF_ENABLE_ONEDNN_OPTS=1
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export USE_OPENMP=1
export USE_AVX512=1
export IPEX_ONEDNN_LAYOUT=1
export PYTORCH_TENSOREXPR=0

i=0
for f in `ls ${input_dir}|grep ${suffix}`; do
  fpath=${input_dir}/${f}
  ((lo=$i*$core_per_instance))
  ((hi=($i+1)*$core_per_instance-1))
  ((m=$i/($n_sample/2))) # $i is index of sample, not index of core!!!!
  ((ncpu=$core_per_instance))
  echo "run modelinfer ${fpath} on core $lo-$hi of socket $m"
	numactl -C $lo-$hi -m $m $script \
		--fasta_paths ${fpath} \
		--output_dir ${out_dir} \
		--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
		--model_names=${model_name} \
    --root_params=${root_params} \
		--uniclust30_database_path=${data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
		--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
		--mgnify_database_path=${data_dir}/mgnify/mgy_clusters_2018_12.fa \
		--pdb70_database_path=${data_dir}/pdb70/pdb70 \
		--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
		--data_dir=${data_dir} \
		--max_template_date=2020-05-14 \
		--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
		--hhblits_binary_path=`which hhblits` \
		--hhsearch_binary_path=`which hhsearch` \
		--jackhmmer_binary_path=`which jackhmmer` \
		--kalign_binary_path=`which kalign` \
		> ${log_dir}/${f}_${n_sample}_${model_name}.txt 2>&1 &
	((i=${i}+1))
done

### wait until done
while true; do
  if [ `ls ${out_dir}|grep timmer|wc -l` -gt 1 ]; then
    n=0
		for f in `ls ${out_dir}|grep timmer`; do
		  status=`cat ${out_dir}/${f}|grep model_infer|wc -l`
			if [[ $status == 1 ]]; then
			  ((n+=1))
			fi
		done
  else
    n=0
  fi
  if [ $n -eq $n_sample ]; then
    break
  fi
  sleep 60
done
echo "iaf2_modelinfer_done"
cd $workdir
