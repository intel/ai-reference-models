# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import pickle
import time
from runners.timmer import Timmers
from runners.saver import load_feature_dict_if_exist

from absl import app, flags, logging
import sys
sys.append(os.path.join(os.path.dirname(__file__), 'alphafold'))
from alphafold.common import protein
from alphafold.relax import relax
import numpy as np
import jax

### Define Flags
flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_string('root_params', None, 'root directory of model parameters') ### updated
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', '/usr/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', '/usr/bin/hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', '/usr/bin/hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/usr/bin/kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('preset', 'full_dbs',
                  ['reduced_dbs', 'full_dbs', 'casp14'],
                  'Choose preset model configuration - no ensembling and '
                  'smaller genetic database config (reduced_dbs), no '
                  'ensembling and full genetic database config  (full_dbs) or '
                  'full genetic database config and 8 model ensemblings '
                  '(casp14).')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('n_cpu', None, 'CPU physical cores used in MSA '
                    'It is dependent on the instance number you want to run '
                    'simultaneosly. e.g. your #CPU_core=32 & #instance=8, '
                    'choose 4', lower_bound=1, required=True)
FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


### helper func: validate required options
def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


### main func for model inference
def amber_relax(
    timmer: Timmers,
    fasta_name: str,
    output_dir_base: str,
    amber_relaxer: relax.AmberRelaxation):
  print('### Validate preprocessed results.')
  timings = {}
  t0_total = time.time()
  output_dir = os.path.join(output_dir_base, fasta_name)
  assert os.path.isdir(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  tmp_output_dir = os.path.join(output_dir, 'intermediates')
  assert os.path.isdir(msa_output_dir)
  assert os.path.isdir(tmp_output_dir)
  ftmp_processed_featdict = os.path.join(
    tmp_output_dir, 
    'processed_features.npz')
  processed_feature_dict = load_feature_dict_if_exist(
    ftmp_processed_featdict)
  processed_feature_dict = jax.tree_map(
    lambda x:np.array(x), processed_feature_dict)
  if processed_feature_dict is None:
    raise FileNotFoundError(
      'Invalid processed features: ',
      ftmp_processed_featdict)
  
  model_name = FLAGS.model_names[0]
  result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
  with open(result_output_path, 'rb') as f:
    prediction_result = pickle.load(f)
  prediction_result = jax.tree_map(
    lambda x:np.array(x), prediction_result)

  print('### load unrelaxed structure')
  unrelaxed_protein = protein.from_prediction(
    processed_feature_dict,
    prediction_result)

  print('### post-adjust: amber-relax')
  relaxed_pdbs = {}
  t_0 = time.time()
  timmer_name = 'amberrelax_%s_from_%s' % (fasta_name, model_name)
  timmer.add_timmer(timmer_name)
  t1_amber = time.time()
  relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
  t2_amber = time.time()
  print('  # [TIME] amber process =', (t2_amber-t1_amber),'sec')
  relaxed_pdbs[model_name] = relaxed_pdb_str
  f_relaxed_output = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
  with open(f_relaxed_output, 'w') as h:
    h.write(relaxed_pdb_str)
  timings[f'relax_{model_name}'] = time.time() - t_0
  timmer.end_timmer(timmer_name)
  timmer.save()
  t_diff = time.time() - t0_total
  timings[f'predict_and_compile_all_models'] = t_diff


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many cml args.')
  use_small_bfd = FLAGS.preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', FLAGS.preset,
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)
  _check_flag('uniclust30_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)
  
  print('### start script for model infer.')
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  # init timmers
  f_timmer = os.path.join(FLAGS.output_dir, 'timmers_%s.txt' % fasta_names[0])
  h_timmer = Timmers(f_timmer)
  print('### use %d CPU cores' % FLAGS.n_cpu)
  # init amber
  h_timmer.add_timmer('amber_relaxation')
  amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)
  h_timmer.end_timmer('amber_relaxation')
  h_timmer.save()
  # init randomizer
  random_seed = FLAGS.random_seed
  if random_seed is None:
    #random_seed = random.randrange(sys.maxsize)
    random_seed = 5582232524994481130
  logging.info('Using random seed %d for the data pipeline', random_seed)
  ### predict
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    h_timmer.add_timmer('predict_%s' % fasta_name)
    amber_relax(
      timmer=h_timmer,
      fasta_name=fasta_name,
      output_dir_base=FLAGS.output_dir,
      amber_relaxer=amber_relaxer)
    h_timmer.end_timmer('predict_%s' % fasta_name)
    h_timmer.save()


if __name__ == '__main__':
  logging.set_verbosity(logging.FATAL)
  flags.mark_flags_as_required([
    'fasta_paths',
    'output_dir',
    'model_names',
    'root_params',
    'data_dir',
    'preset',
    'uniref90_database_path',
    'mgnify_database_path',
    'pdb70_database_path',
    'template_mmcif_dir',
    'max_template_date',
    'obsolete_pdbs_path',
    'n_cpu'
  ])
  app.run(main)
