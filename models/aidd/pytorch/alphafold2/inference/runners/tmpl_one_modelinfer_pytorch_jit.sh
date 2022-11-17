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
data_dir=$root_data
log_dir=$root_home/logs
f_fasta_list=`ls $input_dir|grep ".fa$"`
f_fasta=${f_fasta_list[0]}
n_sample=1
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((core_per_instance=$n_core*$n_socket/$n_sample))
((n_sample_0=$n_sample-1))
script="python run_modelinfer_pytorch_jit.py"
root_params=$root_home/weights/extracted/${model_name}

export LD_PRELOAD=$root_condaenv/lib/libiomp5.so:$root_condaenv/lib/libjemalloc.so:$LD_PRELOAD
export KMP_AFFINITY=granularity=fine,compact,1,0 # 
export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export OMP_NUM_THREADS=$core_per_instance
export TF_ENABLE_ONEDNN_OPTS=1
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export USE_OPENMP=1
export USE_AVX512=1
export IPEX_ONEDNN_LAYOUT=1
export PYTORCH_TENSOREXPR=0
export CUDA_VISIBLE_DEVICES=-1

lo=0
((hi=$core_per_instance-1))
ncpu=$core_per_instance
echo modelinfer ${input_dir}/${f_fasta} on core $lo-$hi of socket 0-1
numactl -C $lo-$hi -m 0,1 $script \
  --n_cpu $core_per_instance \
	--fasta_paths ${input_dir}/${f_fasta} \
	--output_dir ${out_dir} \
	--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
	--model_names=${model_name} \
  --root_params=${root_params} \
	--uniclust30_database_path=${data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
	--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
	--mgnify_database_path=${data_dir}/mgnify/mgy_clusters.fa \
	--pdb70_database_path=${data_dir}/pdb70/pdb70 \
	--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
	--data_dir=${data_dir} \
	--max_template_date=2020-05-14 \
	--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
	--hhblits_binary_path=`which hhblits` \
	--hhsearch_binary_path=`which hhsearch` \
	--jackhmmer_binary_path=`which jackhmmer` \
	--kalign_binary_path=`which kalign`
