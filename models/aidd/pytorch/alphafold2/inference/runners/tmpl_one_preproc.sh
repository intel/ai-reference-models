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
log_dir=$root_home/logs # root of logs
n_sample=1
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((n_sample_0=$n_sample-1))
((core_per_instance=$n_core*$n_socket/$n_sample))
script="python run_preprocess.py"

dir_input=$root_home/samples
f_list=`ls $dir_input |grep ".fa$"`
f=${f_list[0]}
lo=0
((hi=$core_per_instance-1))
((ncpu=$core_per_instance))

echo preprocessing ${f} on cores $lo to $hi on full sockets
numactl -C $lo-$hi -m 0,1 $script \
  --n_cpu $ncpu \
	--fasta_paths ${f} \
	--output_dir ${out_dir} \
	--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
	--model_names=model_1 \
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
