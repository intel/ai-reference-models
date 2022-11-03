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
log_dir=${root_home}/logs
f_fasta_list=`ls ${root_home}/samples |grep ".fa$"`
f_fasta=${f_fasta_list[0]}
n_sample=1
script='run_amber.py'
root_params=${root_home}/weights/extracted/${model_name}

echo modelinfer ${input_dir}/${f_fasta}
python $script \
  --n_cpu 8 \
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
	--kalign_binary_path=`which kalign` \
