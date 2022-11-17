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
import json
import os
import pathlib
import pickle
import time
from runners.timmer import Timmers
from runners.saver import load_feature_dict_if_exist

from absl import app
from absl import flags
from absl import logging
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'alphafold'))
from alphafold.common import protein, confidence
from alphafold_pytorch_jit import config
from alphafold_pytorch_jit import net as model
import numpy as np
import jax
import intel_extension_for_pytorch as ipex

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
def alphafold_infer(
    timmer: Timmers,
    fasta_name: str,
    output_dir_base: str,
    random_seed: int):
  print('### Validate preprocessed results.')
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  print("### [INFO] output_dir=", output_dir)
  assert os.path.isdir(output_dir)
  tmp_output_dir = os.path.join(output_dir, 'intermediates')
  #assert os.path.isdir(msa_output_dir)
  print("#########", tmp_output_dir)
  assert os.path.isdir(tmp_output_dir)
  ftmp_processed_featdict = os.path.join(
    tmp_output_dir, 
    'processed_features.npz')
  processed_feature_dict = load_feature_dict_if_exist(
    ftmp_processed_featdict)
  if processed_feature_dict is None:
    raise FileNotFoundError(
      'Invalid processed features: ',
      ftmp_processed_featdict)
  
  t0_total = time.time()
  plddts, relaxed_pdbs = {}, {}
  
  ### prepare model runners
  processed_feature_dict = jax.tree_map(
    lambda x:torch.tensor(x), processed_feature_dict)
  if FLAGS.preset in ('reduced_dbs', 'full_dbs'):
    num_ensemble = 1
  elif FLAGS.preset == 'casp14':
    num_ensemble = 8
  model_runners = {}
  for model_name in FLAGS.model_names:
    model_config = config.model_config(model_name)
    model_config['data']['eval']['num_ensemble'] = num_ensemble
    root_params = FLAGS.root_params
    model_runner = model.RunModel(
      model_config, 
      root_params, 
      timmer,
      random_seed)
    model_runners[model_name] = model_runner 
    model_runners[model_name].eval()
    model_runners[model_name] = ipex.optimize(model_runners[model_name])

  for model_name, model_runner in model_runners.items():
    print('### [INFO] Execute model inference')
    timmer_name = f'model inference: {model_name}'
    timmer.add_timmer(timmer_name)
    with torch.no_grad():
      prediction_result = model_runner(processed_feature_dict)
    processed_feature_dict = jax.tree_map(
      lambda x:x.detach().numpy(),
      processed_feature_dict)
    timmer.end_timmer(timmer_name)
    timmer.save()

    print('### [INFO] post-assessment: plddt')
    timmer_name = f'post-assessment by plddt: {model_name}'
    timmer.add_timmer(timmer_name)
    plddts[model_name] = np.mean(prediction_result['plddt'])
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)
    timmer.end_timmer(timmer_name)
    timmer.save()

    print('### [INFO] post-save: unrelaxed structure')
    timmer_name = f'post-save of unrelaxed pdb: {model_name}'
    timmer.add_timmer(timmer_name)
    unrelaxed_protein = protein.from_prediction(
      processed_feature_dict,
      prediction_result)
    unrelaxed_pdb_path = os.path.join(
      output_dir,
      f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as h:
      h.write(protein.to_pdb(unrelaxed_protein))
    timmer.end_timmer(timmer_name)
    timmer.save()
    f_timings_output = os.path.join(output_dir, 'timings.json')
    with open(f_timings_output, 'w') as h:
      h.write(json.dumps(timings, indent=4))


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
  
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  # init timmers
  f_timmer = os.path.join(FLAGS.output_dir, 'timmers_%s.txt' % fasta_names[0])
  h_timmer = Timmers(f_timmer)
  # init amber
  h_timmer.add_timmer('amber_relaxation')
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
    alphafold_infer(
      timmer=h_timmer,
      fasta_name=fasta_name,
      output_dir_base=FLAGS.output_dir,
      random_seed=random_seed)
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
    'obsolete_pdbs_path'
  ])
  app.run(main)
