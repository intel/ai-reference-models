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

"""Full AlphaFold protein structure prediction script."""
import os
import pathlib
import time
from typing import Dict
from runners.timmer import Timmers
from runners.saver import save_feature_dict, load_feature_dict_if_exist

from absl import app, flags, logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'alphafold')) # import original AlphaFold2
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

# Internal import (7716).

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
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


def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


def predict_structure(
    timmer: Timmers,
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  tmp_output_dir = os.path.join(output_dir, 'intermediates')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)
  if not os.path.exists(tmp_output_dir):
    os.makedirs(tmp_output_dir)

  is_save_intermediates = True
  # Get features.
  t_0 = time.time()
  timmer.add_timmer('predict_%s_datapipeline' % fasta_name)
  ftmp_featdict = os.path.join(tmp_output_dir, 'features.npz')
  feature_dict = load_feature_dict_if_exist(ftmp_featdict)
  if feature_dict is None:
    print('#### 1. start data pipeline preprocessing from de novo.')
    feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir)
    if is_save_intermediates:
      save_feature_dict(ftmp_featdict, feature_dict)
  else:
    print('==== 1. loaded archive of data pipeline preprocessing.')
  timings['features'] = time.time() - t_0
  timmer.end_timmer('predict_%s_datapipeline' % fasta_name)
  timmer.save()

  # Run the models.
  for model_name, model_runner in model_runners.items():
    logging.info('Running model %s', model_name)
    t_0 = time.time()
    timmer.add_timmer('processfeatures_%s_by_%s' % (fasta_name, model_name))
    ftmp_processed_featdict = os.path.join(tmp_output_dir, 'processed_features.npz')
    processed_feature_dict = load_feature_dict_if_exist(ftmp_processed_featdict)
    if processed_feature_dict is None:
      print('#### 2. start feature pre-model processing from de novo.')
      processed_feature_dict = model_runner.process_features(
        feature_dict, 
        random_seed=random_seed
      )
      if is_save_intermediates:
        save_feature_dict(ftmp_processed_featdict, processed_feature_dict)
    else:
      print('==== 2. loaded archive of feature pre-model processing.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  use_small_bfd = FLAGS.preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', FLAGS.preset,
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)
  _check_flag('uniclust30_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)

  if FLAGS.preset in ('reduced_dbs', 'full_dbs'):
    num_ensemble = 1
  elif FLAGS.preset == 'casp14':
    num_ensemble = 8

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  
  # init timmers
  f_timmer = os.path.join(FLAGS.output_dir, 'timmers_%s.txt' % fasta_names[0])
  h_timmer = Timmers(f_timmer)

  h_timmer.add_timmer('template_hit_featurizer')
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  h_timmer.end_timmer('template_hit_featurizer')
  h_timmer.save()
  h_timmer.add_timmer('data_pipeline')
  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd)
  h_timmer.end_timmer('data_pipeline')
  h_timmer.save()

  model_runners = {}
  for model_name in FLAGS.model_names:
    h_timmer.add_timmer('model_%s_compilation' % model_name)
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner
    h_timmer.end_timmer('model_%s_compilation' % model_name)
  h_timmer.save()

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  #random_seed = FLAGS.random_seed
  random_seed = 5582232524994481130
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    h_timmer.add_timmer('predict_%s' % fasta_name)
    predict_structure(
        timmer=h_timmer,
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        random_seed=random_seed)
    h_timmer.end_timmer('predict_%s' % fasta_name)
    h_timmer.save()


if __name__ == '__main__':
  logging.set_verbosity(logging.FATAL)
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'model_names',
      'data_dir',
      'preset',
      'uniref90_database_path',
      'mgnify_database_path',
      'pdb70_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path'
  ])
  t1 = time.time()
  app.run(main)
  t2 = time.time()
  print('### total time: %d sec' % (t2-t1))
