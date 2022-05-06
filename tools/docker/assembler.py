# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multipurpose TensorFlow Docker Helper.

- Assembles Dockerfiles
- Builds images (and optionally runs image tests)
- Pushes images to Docker Hub (provided with credentials)

Logs are written to stderr; the list of successfully built images is
written to stdout.

Read README.md (in this directory) for instructions!
"""

import collections
import copy
from distutils.dir_util import copy_tree
import errno
import glob
import itertools
import json
import multiprocessing
import os
import platform
import re
import shutil
import sys
import urllib
import subprocess

from absl import app
from absl import flags
import cerberus
import docker
import yaml
import tarfile
import tempfile

FLAGS = flags.FLAGS

flags.DEFINE_string('hub_username', None,
                    'Dockerhub username, only used with --upload_to_hub')

flags.DEFINE_string(
    'hub_password', None,
    ('Dockerhub password, only used with --upload_to_hub. Use from an env param'
     ' so your password isn\'t in your history.'))

flags.DEFINE_integer('hub_timeout', 3600,
                     'Abort Hub upload if it takes longer than this.')

flags.DEFINE_string(
    'repository', 'tensorflow',
    'Tag local images as {repository}:tag (in addition to the '
    'hub_repository, if uploading to hub)')

flags.DEFINE_string(
    'hub_repository', None,
    'Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')

flags.DEFINE_boolean(
    'upload_to_hub',
    False,
    ('Push built images to Docker Hub (you must also provide --hub_username, '
     '--hub_password, and --hub_repository)'),
    short_name='u',
)

flags.DEFINE_boolean(
    'print_models', False, 'print out the list of models', short_name='t')

flags.DEFINE_boolean(
    'build_packages', False, 'Do not build packages', short_name='z')

flags.DEFINE_boolean(
    'construct_dockerfiles', False, 'Do not build Dockerfiles', short_name='d')

flags.DEFINE_boolean(
    'generate_deployments', False, 'Do not generate deployments')

flags.DEFINE_boolean(
    'generate_documentation', False, 'Do not create README.md', short_name='e')

flags.DEFINE_boolean(
    'generate_deployments_tests', False, 'Do not create model-builder.bats for generate-deployment tests')

flags.DEFINE_boolean(
    'keep_temp_dockerfiles',
    False,
    'Retain .temp.Dockerfiles created while building images.',
    short_name='k')

flags.DEFINE_boolean(
    'build_images', False, 'Do not build images', short_name='b')

flags.DEFINE_boolean(
    'list_images', False, 'Do not list images that would be built', short_name='c')

flags.DEFINE_boolean(
    'list_packages', False, 'Do not list model packages that would be built')

flags.DEFINE_string(
    'run_tests_path', None,
    ('Execute test scripts on generated Dockerfiles before pushing them. '
     'Flag value must be a full path to the "tests" directory, which is usually'
     ' $(realpath ./tests). A failed tests counts the same as a failed build.'))

flags.DEFINE_boolean(
    'stop_on_failure', False,
    ('Stop processing tags if any one build fails. If False or not specified, '
     'failures are reported but do not affect the other images.'))

flags.DEFINE_boolean(
    'dry_run',
    False,
    'Do not build or deploy anything at all.',
    short_name='n',
)

flags.DEFINE_string(
    'exclude_tags_matching',
    None,
    ('Regular expression that skips processing on any tag it matches. Must '
     'match entire string, e.g. ".*gpu.*" ignores all GPU tags.'),
    short_name='x')

flags.DEFINE_multi_string(
    'only_tags_matching',
    [],
    ('Regular expression that skips processing on any tag it does not match. '
     'Must match entire string, e.g. ".*gpu.*" includes only GPU tags.'),
    short_name='i')

flags.DEFINE_string(
    'model_dir',
    '.', 'Path to the model repo.')

flags.DEFINE_string(
    'output_dir',
    'output', 'Path to an output directory for model packages.'
    ' Will be created if it doesn\'t exist.')

flags.DEFINE_string(
    'dockerfile_dir',
    'models/dockerfiles', 'Path to an output directory for Dockerfiles.'
    ' Will be created if it doesn\'t exist.'
    ' Existing files in this directory will be deleted when new Dockerfiles'
    ' are made.',
    short_name='o')

flags.DEFINE_string(
    'deployment_dir',
    'models/deployments', 'Path to directory for k8 deployments.'
    ' Will be created if it doesn\'t exist.'
    ' Existing files under this directory will be deleted when new deployments'
    ' are created.')

flags.DEFINE_string(
    'partial_dir',
    './partials',
    'Path to a directory containing foo.partial.Dockerfile partial files.'
    ' can have subdirectories, e.g. "bar/baz.partial.Dockerfile".',
    short_name='p')

flags.DEFINE_multi_string(
    'release', [],
    'Set of releases to build and tag. Defaults to every release type.',
    short_name='r')

flags.DEFINE_multi_string(
    'arg', [],
    ('Extra build arguments. These are used for expanding tag names if needed '
     '(e.g. --arg _TAG_PREFIX=foo) and for using as build arguments (unused '
     'args will print a warning).'),
    short_name='a')

flags.DEFINE_boolean(
    'nocache', False,
    'Disable the Docker build cache; identical to "docker build --no-cache"')

flags.DEFINE_string(
    'spec_dir',
    './specs',
    'Path to the YAML specification directory',
    short_name='s')

flags.DEFINE_string(
    'framework',
    'tensorflow',
    'Name of the deep learning framework. This is being used with the '
    '--generate_new_spec arg to map the model zoo directory structure, which '
    'includes a directory for the framework.')

flags.DEFINE_string(
    'use_case',
    '',
    'Name of the use_case for when generating a new spec. This is used in the '
    'case where the model\'s folders don\'t already exist, so the directory '
    'structure cannot be used to infer the use case.')

flags.DEFINE_string(
    'device',
    'cpu',
    'Name of the device folder to be used.'
)

flags.DEFINE_string(
    'generate_new_spec',
    None,
    'Used to auto generate a spec with model package files. Specify the name '
    'for the new spec, which should be formatted like modelname-precision-mode')

flags.DEFINE_string(
    'model_download',
    None,
    'Use only with --generate_new_spec to specify the URL to download a '
    'pretrained model.')

flags.DEFINE_boolean(
    'verbose', False,
    'verbose mode')


# Schema to verify the contents of merged spec yaml with Cerberus.
# Must be converted to a dict from yaml to work.
# Note: can add python references with e.g.
# !!python/name:builtins.str
# !!python/name:__main__.funcname
# (but this may not be considered safe?)
SCHEMA_TEXT = """
header:
  type: string

slice_sets:
  type: dict
  keyschema:
    type: string
  valueschema:
    type: list
    schema:
      type: dict
      schema:
        add_to_name:
          type: string
        dockerfile_exclusive_name:
          type: string
        dockerfile_subdirectory:
          type: string
        partials:
          type: list
          schema:
            type: string
            ispartial: true
        documentation:
          type: list
          default: []
          schema:
            type: dict
            schema:
              name:
                type: string
              uri:
                type: string
              text_replace:
                type: dict
              docs:
                type: list
                schema:
                  type: dict
                  schema:
                    name:
                      type: string
                    uri:
                      type: string
        test_runtime:
          type: string
          required: false
        tests:
          type: list
          default: []
          schema:
            type: string
        args:
          type: list
          default: []
          schema:
            type: string
            isfullarg: true
        files:
          type: list
          schema:
            type: dict
            schema:
              source:
                type: string
              destination:
                type: string
        wrapper_package_files:
          type: list
          schema:
            type: dict
            schema:
              source:
                type: string
              destination:
                type: string
        downloads:
          type: list
          schema:
            type: dict
            schema:
              source:
                type: string
              destination:
                type: string
        runtime:
          type: dict
          schema:
            command:
              type: list
              default: []
              schema:
                type: string
            args:
              type: list
              default: []
              schema:
                type: string
            env:
              type: list
              schema:
                type: dict
                schema:
                  name:
                    type: string
                  value:
                    type: string
            resources:
              type: list
              schema:
                type: dict
                schema:
                  name:
                    type: string
                  value:
                    type: string
            tolerations:
              type: list
              schema:
                type: dict
                schema:
                  name:
                    type: string
                  value:
                    type: string
            tests:
              type: list
              schema:
                type: dict
                schema:
                  uri:
                    type: string
                  args:
                    type: list
                    schema:
                      type: dict
                      schema:
                        name:
                          type: string
                        value:
                          type: string

releases:
  type: dict
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      is_dockerfiles:
        type: boolean
        required: false
        default: false
      upload_images:
        type: boolean
        required: false
        default: true
      tag_specs:
        type: list
        required: true
        schema:
          type: string
"""

# Template used when generating new model spec files with a slice set
model_spec_template = {
    'releases':
        {
            'versioned': {'tag_specs': []}
        },
    'slice_sets': {}
}

# Slice set template used when generating new model spec files
slice_set_template = [
    {
        'add_to_name': '',
        'dockerfile_subdirectory': 'model_containers',
        'partials': ['model_package', 'entrypoint'],
        'documentation': [{'text_replace': {}}],
        'args': [],
        'files': [],
        'downloads': []
    }]

class TfDockerTagValidator(cerberus.Validator):
  """Custom Cerberus validator for TF tag spec.

  Note: Each _validate_foo function's docstring must end with a segment
  describing its own validation schema, e.g. "The rule's arguments are...". If
  you add a new validator, you can copy/paste that section.
  """

  def __init__(self, *args, **kwargs):
    # See http://docs.python-cerberus.org/en/stable/customize.html
    if 'partials' in kwargs:
      self.partials = kwargs['partials']
    super(cerberus.Validator, self).__init__(*args, **kwargs)

  def _validate_ispartial(self, ispartial, field, value):
    """Validate that a partial references an existing partial spec.

    Args:
      ispartial: Value of the rule, a bool
      field: The field being validated
      value: The field's value
    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if ispartial and value not in self.partials:
      self._error(field,
                  '{} is not present in the partials directory.'.format(value))

  def _validate_isfullarg(self, isfullarg, field, value):
    """Validate that a string is either a FULL=arg or NOT.

    Args:
      isfullarg: Value of the rule, a bool
      field: The field being validated
      value: The field's value
    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if isfullarg and '=' not in value:
      self._error(field, '{} should be of the form ARG=VALUE.'.format(value))
    if not isfullarg and '=' in value:
      self._error(field, '{} should be of the form ARG (no =).'.format(value))


def eprint(*args, **kwargs):
  if "verbose" in kwargs.keys():
    verbose = kwargs["verbose"]
    del kwargs["verbose"]
    if verbose == True:
      print(*args, file=sys.stderr, flush=True, **kwargs)
  else:
      print(*args, file=sys.stderr, flush=True, **kwargs)


def aggregate_all_slice_combinations(spec, slice_set_names):
  """Figure out all of the possible slice groupings for a tag spec."""
  slice_sets = copy.deepcopy(spec['slice_sets'])

  for name in slice_set_names:
    for slice_set in slice_sets[name]:
      slice_set['set_name'] = name

  slices_grouped_but_not_keyed = [slice_sets[name] for name in slice_set_names]
  all_slice_combos = list(itertools.product(*slices_grouped_but_not_keyed))
  return all_slice_combos


def build_name_from_slices(format_string, slices, args, is_dockerfile=False):
  """Build the tag name (cpu-devel...) from a list of slices."""
  name_formatter = copy.deepcopy(args)
  # don't include the tag prefix in the dockerfile name
  if is_dockerfile:
    name_formatter['_TAG_PREFIX'] = ''
  name_formatter.update({s['set_name']: s['add_to_name'] for s in slices})
  name_formatter.update({
      s['set_name']: s['dockerfile_exclusive_name']
      for s in slices
      if is_dockerfile and 'dockerfile_exclusive_name' in s
  })
  name = format_string.format(**name_formatter)
  return name


def update_args_dict(args_dict, updater):
  """Update a dict of arg values with more values from a list or dict."""
  if isinstance(updater, list):
    for arg in updater:
      key, sep, value = arg.partition('=')
      if sep == '=':
        args_dict[key] = value
  if isinstance(updater, dict):
    for key, value in updater.items():
      args_dict[key] = value
  return args_dict


def get_slice_sets_and_required_args(slice_sets, tag_spec):
  """Extract used-slice-sets and required CLI arguments from a spec string.

  For example, {FOO}{bar}{bat} finds FOO, bar, and bat. Assuming bar and bat
  are both named slice sets, FOO must be specified on the command line.

  Args:
     slice_sets: Dict of named slice sets
     tag_spec: The tag spec string, e.g. {_FOO}{blep}

  Returns:
     (used_slice_sets, required_args), a tuple of lists
  """
  required_args = []
  used_slice_sets = []

  extract_bracketed_words = re.compile(r'\{([^}]+)\}')
  possible_args_or_slice_set_names = extract_bracketed_words.findall(tag_spec)
  for name in possible_args_or_slice_set_names:
    if name in slice_sets:
      used_slice_sets.append(name)
    else:
      required_args.append(name)

  return (used_slice_sets, required_args)


def gather_tag_args(slices, cli_input_args, required_args):
  """Build a dictionary of all the CLI and slice-specified args for a tag."""
  args = {}

  for s in slices:
    args = update_args_dict(args, s['args'])

  args = update_args_dict(args, cli_input_args)
  for arg in required_args:
    if arg not in args:
      eprint(('> Error: {} is not a valid slice_set, and also isn\'t an arg '
              'provided on the command line. If it is an arg, please specify '
              'it with --arg. If not, check the slice_sets list.'.format(arg)))
      exit(1)

  return args


def gather_slice_list_items(slices, key):
  """For a list of slices, get the flattened list of all of a certain key."""
  return list(itertools.chain(*[s[key] for s in slices if key in s]))


def find_first_slice_value(slices, key):
  """For a list of slices, get the first value for a certain key."""
  for s in slices:
    if key in s and s[key] is not None:
      return s[key]
  return None


def assemble_tags(spec, cli_args, enabled_releases, all_partials):
  """Gather all the tags based on our spec.

  Args:
    spec: Nested dict containing full Tag spec
    cli_args: List of ARG=foo arguments to pass along to Docker build
    enabled_releases: List of releases to parse. Empty list = all
    all_partials: Dict of every partial, for reference

  Returns:
    Dict of tags and how to build them
  """
  tag_data = collections.defaultdict(list)

  for name, release in spec['releases'].items():
    for tag_spec in release['tag_specs']:
      if enabled_releases and name not in enabled_releases:
        eprint(('> Skipping release {}'.format(name)), verbose=FLAGS.verbose)
        continue

      used_slice_sets, required_cli_args = get_slice_sets_and_required_args(
          spec['slice_sets'], tag_spec)

      slice_combos = aggregate_all_slice_combinations(spec, used_slice_sets)
      for slices in slice_combos:

        tag_args = gather_tag_args(slices, cli_args, required_cli_args)
        tag_name = build_name_from_slices(
            tag_spec, slices, tag_args, is_dockerfile=False)
        dockerfile_tag_name = build_name_from_slices(
            tag_spec, slices, tag_args, is_dockerfile=True)
        used_partials = gather_slice_list_items(slices, 'partials')
        used_tests = gather_slice_list_items(slices, 'tests')
        documentation = []
        if 'documentation' in slices[len(slices)-1]:
            documentation = slices[len(slices)-1]['documentation']
            documentation = merge_docs(documentation)
        files_list = gather_slice_list_items(slices, 'files')
        wrapper_files_list = gather_slice_list_items(slices, 'wrapper_package_files')
        downloads_list = gather_slice_list_items(slices, 'downloads')
        test_runtime = find_first_slice_value(slices, 'test_runtime')
        dockerfile_subdirectory = find_first_slice_value(
            slices, 'dockerfile_subdirectory')
        dockerfile_contents = merge_partials(spec['header'], used_partials,
                                             all_partials)
        runtime = {}
        if 'runtime' in slices[len(slices)-1]:
            runtime = slices[len(slices)-1]['runtime']
            env_list = gather_slice_list_items([runtime], 'env')
            runtime.update({ 'env': env_list })

        tag_data[tag_name].append({
            'release': name,
            'tag_spec': tag_spec,
            'is_dockerfiles': release['is_dockerfiles'],
            'upload_images': release['upload_images'],
            'cli_args': tag_args,
            'dockerfile_subdirectory': dockerfile_subdirectory or '',
            'partials': used_partials,
            'tests': used_tests,
            'test_runtime': test_runtime,
            'dockerfile_contents': dockerfile_contents,
            'files': files_list,
            'wrapper_package_files': wrapper_files_list,
            'downloads': downloads_list,
            'documentation': documentation,
            'dockerfile_tag_name': dockerfile_tag_name,
            'runtime': runtime,
        })

  return tag_data


def merge_partials(header, used_partials, all_partials):
  """Merge all partial contents with their header."""
  used_partials = list(used_partials)
  return '\n'.join([header] + [all_partials[u] for u in used_partials])

def doc_contents(path):
  """
  Read document and return contents
    Args:
      path (string): read partials from this directory.
    Returns:
      contents of path.
  """
  contents = ""
  try:
      with open(path, 'r', encoding="utf-8") as f:
          contents = f.read()
  except Exception as e:
    eprint("error reading {} exception: {}".format(path, e))
    raise e
  return contents

def merge_docs(docs_list, package_type='model'):
  """Build the documents and fills in the contents for each doc entry"""
  for doc in docs_list:
    contents = ''
    for doc_partial in doc["docs"]:
      uri = ''
      if package_type == 'model' and 'uri' in doc_partial:
        uri = doc_partial['uri']
      if uri:
        contents += '\n'.join([doc_contents(uri) + '\n'])
    doc.update({'contents': contents})
  return docs_list

def upload_in_background(hub_repository, dock, image, tag):
  """Upload a docker image (to be used by multiprocessing)."""
  image.tag(hub_repository, tag=tag)
  eprint(dock.images.push(hub_repository, tag=tag), verbose=FLAGS.verbose)


def mkdir_p(path):
  """Create a directory and its parents, even if it already exists."""
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def delete_dockerfiles(all_tags, dir_path):
    """ Delete dockerfiles based on the directory and specs """
    for tag, tag_defs in all_tags.items():
        for tag_def in tag_defs:
            if tag_def['is_dockerfiles']:
                path = os.path.join(dir_path,
                                    tag_def['dockerfile_subdirectory'],
                                    tag + '.Dockerfile')
                try:
                    if os.path.exists(path):
                      eprint('> Deleting {}'.format(path), verbose=FLAGS.verbose)
                      os.unlink(path)
                except Exception as e:
                    print('Error while deleting dockerfile: {}'.format(path))
                    print(e)


def gather_existing_partials(partial_path):
  """Find and read all available partials.

  Args:
    partial_path (string): read partials from this directory.

  Returns:
    Dict[string, string] of partial short names (like "ubuntu/python" or
      "bazel") to the full contents of that partial.
  """
  partials = {}
  for path, _, files in os.walk(partial_path):
    for name in files:
      fullpath = os.path.join(path, name)
      if '.partial.Dockerfile' not in fullpath:
        eprint(('> Probably not a problem: skipping {}, which is not a '
                'partial.').format(fullpath), verbose=FLAGS.verbose)
        continue
      # partial_dir/foo/bar.partial.Dockerfile -> foo/bar
      simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
      with open(fullpath, 'r', encoding="utf-8") as f:
        try:
          partial_contents = f.read()
        except Exception as e:
          eprint("error reading {} exception: {}".format(simple_name, e))
          sys.exit(1)
      partials[simple_name] = partial_contents
  return partials

def get_package_name(package_def):
  if "cli_args" in package_def:
    cli_args = package_def["cli_args"]
    if "PACKAGE_NAME" in cli_args:
      return cli_args["PACKAGE_NAME"]
  return None

def get_file(source, destination):
    if os.path.isdir(source):
        if not os.path.isdir(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        file_list = copy_tree(source, destination)
        eprint("> Copied {} to {}".format(source, destination), verbose=FLAGS.verbose)
        doc_partials_dir = os.path.join(destination, ".docs")
        if os.path.isdir(doc_partials_dir):
            shutil.rmtree(doc_partials_dir, ignore_errors=True)
        for gitignore in [f for f in file_list if '.gitignore' in f]:
            os.remove(gitignore)
    elif os.path.isfile(source):
        # Ensure that the directories exist first, otherwise the file copy will fail
        if not os.path.isdir(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        shutil.copy(source, destination)
        eprint("> Copied {} to {}".format(source, destination), verbose=FLAGS.verbose)
    else:
        eprint("ERROR: Unable to find file or directory: {}".format(source))
        sys.exit(1)

def get_download(source, destination):
    # Ensure that the directories exist first, otherwise the file copy will fail
    if not os.path.isdir(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    if source.lower().startswith('http'):
        urllib.request.Request(source)
    else:
        raise ValueError from None
    urllib.request.urlretrieve(source, destination)  # nosec
    eprint("Copied {} to {}".format(source, destination), verbose=FLAGS.verbose)

def run(cmd):
    proc = subprocess.Popen(cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr

def set_test_args(tag_def, test, test_uri):
  with open(test_uri, 'r', encoding="utf-8") as f:
    test_file = f.read()
    if 'args' in test:
      for arg in test['args']:
        if arg['value'] and not arg['value'].startswith('$'):
          test_file = test_file.replace('${}'.format(arg['name']), arg['value'])
        else:
          # look up test arg value in env vars
          for env in tag_def['runtime']['env']:
            if env['name'] == arg['name']:
              test_file = test_file.replace(arg['value'], env['value'])
    return test_file

def cfg_set_namespace(package, dir, tag_def):
  for env in tag_def['runtime']['env']:
    if not 'name' in env:
      eprint("name missing in env values for {}".format(package))
      continue
    if not 'value' in env:
      eprint("value missing in env values for {}".format(package))
      continue
    if env['name'] == 'NAME_SPACE':
      cmd = ["kustomize", "edit", "set", 'namespace',  env['value']]
      cwd = os.getcwd()
      os.chdir(dir)
      returncode, out, err = run(cmd)
      if returncode != 0:
        eprint("returncode:{}\nstdout:{}\nstderr:{}".format(returncode, out, err), verbose=FLAGS.verbose)
      os.chdir(cwd)

def cfg_set_values(dir, tag_def):
  if not 'runtime' in tag_def:
    sys.exit("No runtime section with the specification file")
  runtime = tag_def['runtime']
  if not 'env' in runtime:
    sys.exit("No runtime.env section with the specification file")
  for env in runtime['env']:
    if 'name' in env and 'value' in env:
      eprint("name:{} value:{}".format(env['name'], env['value']), verbose=FLAGS.verbose)
      name = env['name']
      value = env['value']
      if name != 'NAME_SPACE':
        cmd = ["kustomize", "cfg", "set", dir, name, value, "--set-by", "model-builder", "-R"]
        returncode, out, err = run(cmd)
        if returncode != 0:
          eprint("returncode:{}\nstdout:{}\nstderr:{}".format(returncode, out, err), verbose=FLAGS.verbose)

def write_deployment(tag_def):
  output_dir = os.path.join(os.getcwd(), FLAGS.output_dir)
  if not os.path.isdir(output_dir):
    sys.exit("You must create the k8s package prior to calling generate-deployment")
  package = get_package_name(tag_def)
  if not package:
    sys.exit("Unable to generate deployment, because the spec doesn't have a package name")
  deployments_top_dir = os.path.join(os.getcwd(), FLAGS.deployment_dir, package)
  if package != None:
    tar_file = os.path.join(FLAGS.output_dir, "{}.tar.gz".format(package))
    try:
      temp_dir = tempfile.mkdtemp()
      extract_tar(tar_file, temp_dir)
      mlops_dir = os.path.join(temp_dir, package, 'quickstart/mlops')
      cfg_set_values(mlops_dir, tag_def)
      _, out, _ = run(["find", mlops_dir, "-name", "kustomization.yaml"])
      kustomization_dirs  = out.splitlines()
      for kustomization_dir in kustomization_dirs:
        dir = os.path.dirname(kustomization_dir).decode("utf-8")
        eprint("dir:{}".format(dir), verbose=FLAGS.verbose)
        m = re.search('^.*mlops\/(.*)$', dir)
        subdir = m.group(1)
        subdir_dirs = subdir.split('/')
        subdir_dirs_root = subdir.split('/')[0]
        if len(subdir_dirs) == 1:
          subdir_dirs_leaf = subdir_dirs_root
        else:
          subdir_dirs_leaf = subdir.split('/')[1]
        deployment_dir = os.path.join(deployments_top_dir, subdir_dirs_root)
        deployment_file = os.path.join(deployment_dir, subdir_dirs_leaf+".yaml")
        if not os.path.isdir(deployment_dir):
          mkdir_p(deployment_dir)
        deployment_output = os.path.join(deployment_dir, deployment_file)
        cfg_set_namespace(package, dir, tag_def)
        returncode, out, err = run(["kustomize", "build", dir])
        eprint("Creating deployment file: {}".format(deployment_output[len(os.getcwd()+"/models/"):]))
        with open(deployment_output, "w", encoding="utf-8") as file:
          file.write(out.decode("utf-8"))
    except Exception as e:
      eprint("Error when generating deployment from k8s package: {}".format(tar_file), verbose=FLAGS.verbose)
      raise e
    finally:
      eprint("Deleting temp directory: {}".format(temp_dir), verbose=FLAGS.verbose)
      shutil.rmtree(temp_dir)

def write_package(package_def, succeeded_packages, failed_packages):
  output_dir = os.path.join(os.getcwd(), FLAGS.output_dir)
  if not os.path.isdir(output_dir):
      eprint(">> Creating directory: {}".format(output_dir), verbose=FLAGS.verbose)
      os.mkdir(output_dir)

  package = get_package_name(package_def)
  if package != None:
    tar_file = os.path.join(FLAGS.output_dir, "{}.tar.gz".format(package))
    eprint("> Creating package: {}".format(tar_file), verbose=FLAGS.verbose)

    try:
      temp_dir = tempfile.mkdtemp()

      # Grab things from the files list
      model_dir=os.path.join(os.getcwd(), FLAGS.model_dir)
      if "files" in package_def.keys():
        for item in package_def["files"]:
          source = os.path.join(model_dir, item["source"])
          destination = os.path.join(temp_dir, item["destination"])
          get_file(source, destination)
      # Grab things from the downloads list
      if "downloads" in package_def.keys():
        for item in package_def["downloads"]:
          source = item["source"]
          destination = os.path.join(temp_dir, item["destination"])
          get_download(source, destination)
      # Write tar file
      eprint("Writing {} to {}".format(temp_dir, tar_file), verbose=FLAGS.verbose)
      with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(temp_dir, arcname=package)
      succeeded_packages.append(tar_file)

      # Create wrapper package (optional)
      wrapper_temp_dir = None
      wrapper_packages_output_dir = os.path.join(FLAGS.output_dir, "wrapper_packages")
      if "wrapper_package_files" in package_def.keys() and \
              len(package_def["wrapper_package_files"]) > 0:
          wrapper_tar_file = os.path.join(wrapper_packages_output_dir,
                                          "{}.tar.gz".format(package))
          wrapper_temp_dir = tempfile.mkdtemp()

          if not os.path.isdir(wrapper_packages_output_dir):
            os.mkdir(wrapper_packages_output_dir)

          # update tar_file string so that the error output is correct in case this fails
          tar_file += "/" + wrapper_tar_file

          for item in package_def["wrapper_package_files"]:
            if not item["source"] and "info.txt" in item["destination"]:
              # Special case to create the info.txt file
              info_file_path = os.path.join(wrapper_temp_dir, item["destination"])
              try:
                model_branch = os.environ["GIT_BRANCH"] if "GIT_BRANCH" in os.environ else ""
                model_commit = os.environ["GIT_COMMIT"] if "GIT_COMMIT" in os.environ else ""

                if not model_branch or not model_commit:
                  from git import Repo
                  repo = Repo("/tf/models").head.reference
                  if not model_branch:
                    model_branch = str(repo.name)
                  if not model_commit:
                    model_commit = str(repo.commit.hexsha)
                  eprint("Write info.txt to {}".format(info_file_path), verbose=FLAGS.verbose)
                with open(info_file_path, "w", encoding="utf-8") as f:
                  f.write("model_zoo_branch: {}\n".format(model_branch))
                  f.write("model_zoo_commit: {}\n".format(model_commit))
              except Exception as e:
                  eprint("Error while creating the info.txt file")
                  eprint(e)
            else:
              source = os.path.join(model_dir, item["source"])
              destination = os.path.join(wrapper_temp_dir, item["destination"])
              get_file(source, destination)

          eprint("Writing {} to {}".format(wrapper_temp_dir, wrapper_tar_file),
                 verbose=FLAGS.verbose)
          with tarfile.open(wrapper_tar_file, "w:gz") as tar:
              tar.add(wrapper_temp_dir, arcname=package)
          succeeded_packages.append(wrapper_tar_file)

    except Exception as e:
      failed_packages.append(tar_file)
      eprint("Error when writing package: {}".format(tar_file))
      eprint(e)
    finally:
      eprint("Deleting temp directory: {}".format(temp_dir), verbose=FLAGS.verbose)
      shutil.rmtree(temp_dir)

      if wrapper_temp_dir:
          eprint("Deleting temp directory: {}".format(wrapper_temp_dir),
                 verbose=FLAGS.verbose)
          shutil.rmtree(wrapper_temp_dir)

def extract_tar(tar_file, temp_dir):
    """ Extract tar.gz """
    if tarfile.is_tarfile(tar_file):
      try:
        with tarfile.open(tar_file, "r:gz") as tar:
          tar.extractall(temp_dir)
      except Exception as e:
        eprint("Error when extracting package: {}".format(tar_file))
        raise e

def read_spec_file(spec_dir, spec_file, tag_spec, replace=False):
    with open(os.path.join(spec_dir, spec_file), 'r', encoding="utf-8") as spec_file:
        try:
            spec_contents = yaml.safe_load(spec_file)
            update_spec(tag_spec, spec_contents, replace)
        except Exception as e:
            eprint("exception in {}: {}".format(spec_file, e))
            raise e

def read_spec_files(spec_dir, tag_spec, replace=False):
    """ Recursively read the spec files into one dict, used for everything """
    for spec_file in os.listdir(spec_dir):
        if os.path.isdir(os.path.join(spec_dir, spec_file)):
            tag_spec = read_spec_files(os.path.join(spec_dir, spec_file), tag_spec, replace)
        else:
            read_spec_file(spec_dir, spec_file, tag_spec, replace)
    return tag_spec

def update_spec(a, b, replace=False):
    """Merge two dictionary specs into one, recursing through any embedded dicts."""
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = update_spec(a.get(k, {}), v, replace)
        elif isinstance(v, list) and k in a:
            if len(v) > 0 and isinstance(v[0], dict):
                if replace is True:
                    a[k] = v
                else:
                    # If a list of dicts is detected for an existing key, reject it
                    # This is a duplicate slice set
                    eprint('Duplicate slice set found for {}'.format(k))
                    exit(1)
            else:
                a[k] = list(set(a[k]).union(v))
        elif k in a and a[k] != v:
            # If a string value for an existing key is being overwritten with
            # a different value, the specs are ambiguous
            eprint('Conflicting values found in spec files for key {}'.format(k))
            eprint('  {} vs. {}'.format(a[k], v))
            exit(1)
        else:
            a[k] = v
    return a


def get_use_case_directory(framework, model_name, precision, mode):
    """
    Searches the model zoo repo to find a matching model to get the use case
    directory.
    Returns the use case string and the model name string, since sometimes the
    model name used in the model zoo will be slightly different than the model
    name in the spec since the spec always uses dashes and the model zoo
    sometimes has underscores.
    """
    zoo_model_name = model_name
    search_path = os.path.join(os.getcwd(), FLAGS.model_dir, '*', '*',
                               framework, zoo_model_name)
    matches = glob.glob(search_path)

    if len(matches) == 0:
        # try replacing - with _ in the model name, and search again
        zoo_model_name = zoo_model_name.replace('-', '_')
        original_search_path = search_path
        search_path = os.path.join(os.getcwd(), FLAGS.model_dir, '*', '*',
                                   framework, zoo_model_name)
        matches = glob.glob(search_path)

        if len(matches) == 0:
            folders_searched = original_search_path.lstrip("{}/".format(os.getcwd()))
            if original_search_path != search_path:
                folders_searched += " or {}".format(search_path.lstrip("{}/".format(os.getcwd())))
            sys.exit('\nNo matching model directory was found for found for {}. \n'
                     'If the model does not exist yet, provide the use case arg '
                     'when calling the model-builder.\nFor example: '
                     'model-builder init-spec -f {} {}-{}-{} <use case>'.
                     format(folders_searched, framework, model_name, precision, mode))

    # use the directory path to find use case (which should be right after framework)
    dir_list = matches[0].split('/')
    use_case = dir_list[dir_list.index(framework) - 1]

    return use_case, zoo_model_name


def get_model_name_directory(framework, model_name):
    """ Attempts to find existing directories for the specified model """

    search_model_name = model_name
    search_path = os.path.join(os.getcwd(), FLAGS.model_dir, '*',
                               '*', framework, search_model_name)
    matches = glob.glob(search_path)

    if len(matches) == 0:
        search_model_name = model_name.replace('-', '_')
        search_path = os.path.join(os.getcwd(), FLAGS.model_dir, '*',
                                   '*', framework, search_model_name)
        matches = glob.glob(search_path)

    if len(matches) > 0:
        return search_model_name

    # If no model name was found for this framework, check if other frameworks
    # have used this model
    if framework != "*":
        zoo_model_name = get_model_name_directory("*", model_name)
    else:
        zoo_model_name = ""

    return zoo_model_name


def auto_generate_package_file_list(framework, use_case, model_name, precision, mode, device):
    """
    Auto-generates the list of model package files for the specified model.
    Files that are included are:
    - benchmarks/common
    - benchmarks/launch_benchmark.py
    - model/common
    - quickstart/common
    - the benchmarks directory for the model/mode/precision
    - README file for the model from the benchmarks directory
    - __init__.py files from the benchmarks directory
    - the models directory for the model/mode/precision, if found
    - extra 'common' folders for the model in the benchmarks and models directories, if found
    - quickstart folder for the model
    """
    model_dir = os.path.join(os.getcwd(), FLAGS.model_dir)

    # common directories/files
    model_package_files = [
        {'source': 'models/common', 'destination': 'models/common'},
        {'source': 'quickstart/common', 'destination': 'quickstart/common'}
    ]

    if framework == "tensorflow":
        model_package_files += [
            {'source': 'benchmarks/common',
             'destination': 'benchmarks/common'},
            {'source': 'benchmarks/launch_benchmark.py',
             'destination': 'benchmarks/launch_benchmark.py'}
        ]

    # benchmarks folder and the README.md
    benchmarks_folder = os.path.join('benchmarks', use_case, framework,
                                    model_name, mode, precision)
    if os.path.exists(benchmarks_folder):
        model_package_files.append({'source': benchmarks_folder,
                                    'destination': benchmarks_folder})
    model_readme = os.path.join('benchmarks', use_case, framework,
                                model_name, 'README.md')
    if os.path.exists(model_readme):
        model_package_files.append({'source': model_readme,
                                    'destination': model_readme})

    # __init__.py files in the benchmarks directory
    path = 'benchmarks'
    if os.path.exists(benchmarks_folder):
        for folder in [use_case, framework, model_name, mode]:
            init_file_path = os.path.join(model_dir, path, folder, '__init__.py')
            if os.path.exists(init_file_path):
                model_package_files.append({'source': os.path.join(path, folder, '__init__.py'),
                                            'destination': os.path.join(path, folder, '__init__.py')})
            path = os.path.join(path, folder)

    # models directory folders (these don't exist for every model - check before appending)
    model_folder = os.path.join('models', use_case, framework, model_name,
                                mode, precision)
    if os.path.exists(os.path.join(model_dir, model_folder)):
        model_package_files.append({'source': model_folder,
                                    'destination': model_folder})
    else:
        # try without the precision folder
        model_folder = os.path.join('models', use_case, framework,
                                    model_name, mode)
        if os.path.exists(os.path.join(model_dir, model_folder)):
            model_package_files.append({'source': model_folder,
                                        'destination': model_folder})

        # try without the mode folder
        model_folder = os.path.join('models', use_case, framework,
                                    model_name, precision)
        if os.path.exists(os.path.join(model_dir, model_folder)):
            model_package_files.append({'source': model_folder,
                                        'destination': model_folder})

    # add the model's quickstart folder
    quickstart_folder = os.path.join('quickstart', use_case, framework,
                                   model_name, mode, device, precision)
    quickstart_folder_full_path = os.path.join(model_dir, quickstart_folder)
    if not os.path.exists(quickstart_folder_full_path):
        os.makedirs(quickstart_folder_full_path)
        template_script = "quickstart_template.sh"
        shutil.copyfile("./{}".format(template_script), os.path.join(
            quickstart_folder_full_path, template_script))
        eprint("Added a template for a quickstart script at: {}\n".format(
            os.path.join(quickstart_folder, template_script)), verbose=FLAGS.verbose)
    model_package_files.append({'source': quickstart_folder,
                                'destination': 'quickstart'})

    # look for extra 'common' folders
    for folder in ['benchmarks', 'models', 'quickstart']:
        # check for a common folder for the model/mode
        common_model_folder = os.path.join(folder, 'use_case', framework,
                                           model_name, mode, 'common')
        if os.path.exists(os.path.join(model_dir, common_model_folder)):
            model_package_files.append({'source': common_model_folder,
                                        'destination': common_model_folder})

        # check for a common folder for the model (without mode)
        common_model_folder = os.path.join(folder, 'use_case', framework,
                                           model_name, 'common')
        if os.path.exists(os.path.join(model_dir, common_model_folder)):
            model_package_files.append({'source': common_model_folder,
                                        'destination': common_model_folder})

    # return the list of dictionaries sorted by 'source'
    return sorted(model_package_files, key=lambda f: f['source'])


def auto_generate_documentation_list(framework, use_case, model_name, precision, mode, device):
    """
    Auto-generates the list of model documentation entries {name: <section>, uri: <uri>} for the specified model.
    Entries that are included are:
      - name: Title
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/title.md
      - name: Description
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/description.md
      - name: Download link
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/download.md
      - name: Datasets
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/datasets.md
      - name: Quick Start Scripts
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/quickstart.md
      - name: Bare Metal
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/baremetal.md
      - name: Docker
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/docker.md
      - name: Kubernetes
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/kubernetes.md
      - name: License link
        uri: models/quickstart/use_case/tensorflow/model_name/mode/device/precision/docs/license.md
    """
    model_dir = os.path.join(os.getcwd(), FLAGS.model_dir)

    # the model's documentation folder
    docs_folder = os.path.join('quickstart', use_case, framework,
                                   model_name, mode, device, precision, ".docs")
    docs_folder_full_path = os.path.join(model_dir, docs_folder)
    if os.path.exists(docs_folder_full_path) == False:
      shutil.copytree("./docs", docs_folder_full_path)
      eprint("> Copied {} to {}".format("./docs", docs_folder_full_path), verbose=FLAGS.verbose)
    markdowns = os.listdir(docs_folder_full_path)

    readme_folder = os.path.join(os.path.basename(model_dir), 'quickstart', use_case, framework,
                                   model_name, mode, device, precision)

    documentation = {"name": "README.md", "uri": readme_folder, "docs": []}
    doc_partials = []
    for markdown in markdowns:
        markdown_source_path = os.path.join(os.getcwd(), "docs", markdown)
        markdown_destination_path = os.path.join(os.path.basename(model_dir), docs_folder, markdown)
        if markdown.endswith('.md'):
          with open(markdown_source_path, 'r', encoding='utf8') as markdown_file:
            try:
              first_line = markdown_file.readline().strip()
              matched = re.search("^.* ([0-9]*)\. (.*) -->$", first_line)
              if matched:
                order = matched.group(1).strip()
                name = matched.group(2).strip()
                doc_partials.append({"name": name, "order": int(order), "uri": markdown_destination_path})
            except Exception as e:
              eprint("exception in {}: {}".format(markdown_source_path, e))
              raise e

    doc_partials_sorted = sorted(doc_partials, key=lambda f: f['order'])
    for doc_partial in doc_partials_sorted:
      if 'name' in doc_partial and 'uri' in doc_partial:
        documentation['docs'].append({"name": doc_partial['name'], "uri": doc_partial['uri']})
    return [documentation]

def generate_doc_text_replace_options(use_case, model_name, precision, mode):
    """
    Returns a dictionary of the default text replace options that are used for
    generating documentation. This is used as the default key/value pairs when

    """
    # Define dictionary of keywords to replace and try to set proper
    # capitalization for common words in model names and precisions. This
    # does not cover the preferred formatting for *ALL* models, but it covers
    # some of the basics. The final model name is defined in the spec yaml.
    formatted_model_name = model_name.title().replace("-", " "). \
        replace("Ssd ", "SSD-"). \
        replace("Resnet", "ResNet"). \
        replace("Mobilenet", "MobileNet"). \
        replace("Densenet", "DenseNet"). \
        replace("Bert", "BERT"). \
        replace("Rfcn", "RFCN"). \
        replace("Gnmt", "GNMT")
    formatted_precision = precision.replace("fp32", "FP32"). \
        replace("int8", "Int8"). \
        replace("bfloat16", "BFloat16")

    return {
        "<model name>": formatted_model_name,
        "<precision>": formatted_precision,
        "<mode>": mode,
        "<package url>": "",
        "<package name>": "{}-{}-{}.tar.gz".format(model_name, precision, mode),
        "<package dir>": "{}-{}-{}".format(model_name, precision, mode),
        "<docker image>": "",
        "<use case>": use_case
    }

def auto_generate_model_spec(spec_name):
    """
    Creates a new spec file for the specified model. The spec_name should be
    formatted like modelname-precision-mode (i.e. resnet50-fp32-inference).

    The spec yaml file will be written to the tools/docker/specs
    directory. If a spec file already exists with the same name, the script
    will exit with an error message.

    This function parses the model name, precision, and mode out of the
    spec_name and then maps that to directories in the model zoo.
    """
    # check if spec file for this model/precision/mode already exists
    if FLAGS.framework == "tensorflow":
        if FLAGS.device == "cpu":
            spec_file_name = '{}_spec.yml'.format(spec_name)
        else:
            spec_file_name = '{}-{}_spec.yml'.format(FLAGS.device, spec_name)
    else:
        spec_file_name = '{}-{}_spec.yml'.format(FLAGS.framework, spec_name)
    spec_file_path = os.path.join(FLAGS.spec_dir, spec_file_name)
    if os.path.isfile(spec_file_path):
        sys.exit('The spec file already exists: {}'.format(spec_file_name))

    # regex to parse out the model name-precision-mode from the spec name
    regex_pattern = r'(\S+)-(\S+)-(\S+)'
    matched_groups = re.findall(regex_pattern, spec_name)[0]

    if len(matched_groups) != 3:
        error_message = 'Unexpected slice name format: {}. Regex couldn\'t parse ' \
                        'out model name, precision, and mode. Expected 3 ' \
                        'groups but found {}'.format(spec_name, len(matched_groups))
        sys.exit(error_message)

    model_name = matched_groups[0]
    precision = matched_groups[1]
    mode = matched_groups[2]
    framework = FLAGS.framework
    device = FLAGS.device

    if not FLAGS.use_case:
        zoo_use_case, zoo_model_name = get_use_case_directory(framework, model_name, precision, mode)
    else:
        zoo_use_case = FLAGS.use_case

        # Look for the model name directory (which might be slightly different
        # than the model_name from the spec name, due to dashes vs underscores)
        zoo_model_name = get_model_name_directory(framework, model_name)

        # If we have no precedent for the model name directory, then use the
        # model_name as it is from the spec name
        if not zoo_model_name:
            zoo_model_name = model_name

    # Create directories if they don't already exist. For TF, we want
    # the `benchmarks` and `models` folder, but for pytorch and other
    # frameworks, just create directories fin the models folder for now
    directories = ["models"]
    if framework == "tensorflow":
        directories += ["benchmarks"]

    for dir in directories:
        dir_path = os.path.join(FLAGS.model_dir, dir, zoo_use_case,
                                framework, zoo_model_name, mode, precision)
        if not os.path.exists(dir_path):
            eprint('\nThe directory \"{}\" does not exist.'.format(dir_path))
            user_response = input('Do you want the model-builder to create it? [y/N] ')

            if user_response.lower() == "y":
                os.makedirs(dir_path)
                eprint("Created directory: {}".format(dir_path))

    use_case_dashes = zoo_use_case.replace('_', '-')

    eprint('\nUse case: {}\nFramework: {}\nModel name: {}\nMode: {}\nPrecision: {}\n'.format(
        use_case_dashes, framework, model_name, mode, precision), verbose=FLAGS.verbose)

    # grab at copy of the model spec template and edit it for this model
    model_spec = model_spec_template.copy()

    if framework == "tensorflow":
        model_spec['releases']['versioned']['tag_specs'] = \
            ['{_TAG_PREFIX}{intel-tf}{' + use_case_dashes + '}{' + spec_name + '}']
    else:
        model_spec['releases']['versioned']['tag_specs'] = \
            ['{_TAG_PREFIX}{' + framework + '}{' + spec_name + '}']

    # grab a copy of the slice set template and edit it for this model
    model_slice_set = slice_set_template.copy()
    model_slice_set[0]['add_to_name'] = '-{}'.format(spec_name)
    model_slice_set[0]['args'].append("PACKAGE_NAME={}".format(spec_name))
    model_slice_set[0]['files'] = auto_generate_package_file_list(
        framework, zoo_use_case, zoo_model_name, precision, mode, device)
    model_slice_set[0]['documentation'] = auto_generate_documentation_list(
        framework, zoo_use_case, zoo_model_name, precision, mode, device)

    # add text replace options for the documentation
    text_replace_dict = generate_doc_text_replace_options(
        zoo_use_case, zoo_model_name, precision, mode)
    model_slice_set[0]['documentation'][0]['text_replace'] = text_replace_dict

    # add a download, if there's one defined
    if FLAGS.model_download:
        model_url = FLAGS.model_download
        model_filename = os.path.basename(model_url)
        model_slice_set[0]['downloads'].append(
            {'source': model_url, 'destination': model_filename})

    # add slice set section
    model_spec['slice_sets'][spec_name] = model_slice_set

    # write the model spec to a file
    with open(spec_file_path, 'w', encoding="utf-8") as f:
        yaml.dump(model_spec, f)

    # print out info for the user to see the spec and file name
    eprint(yaml.dump(model_spec), verbose=FLAGS.verbose)
    eprint("\nWrote the spec file to your directory at "
           "tools/docker/specs/{}/{}\nPlease edit the file if additional "
           "files, partials, or downloads are needed.\n".format(framework, spec_file_name))

    # print out the documentation text_replace options
    eprint("The spec file has documentation text replacement setup for the following key/values:")
    for k, v in text_replace_dict.items():
        eprint("    {}: {}".format(k, v))
    eprint("The text replacement will happen when the README.md is generated "
           "from the doc partials.\nThe key/values can be edited in the spec "
           "yaml file.\n")

    # print out a note about the doc partials
    eprint("Documentation partial files were written to your intelai/models "
           "directory at:\n{}\nPlease edit these files to fill in the "
           "information for your model.\n".format(
               os.path.join('quickstart', zoo_use_case, framework, zoo_model_name,
                            mode, device, precision, ".docs")))

def get_tag_spec(spec_dir, partials):
  """ Reads in a spec files under spec_dir """
  # Read the spec files into one dict, used for everything
  tag_spec = read_spec_files(spec_dir, {})
  spec_home = None
  if 'MODEL_BUILDER_SPEC_HOME' in os.environ:
      spec_home = os.environ['MODEL_BUILDER_SPEC_HOME']
  elif 'HOME' in os.environ:
      spec_home = os.path.join(os.environ['HOME'], ".config", "model-builder", "specs")
  if spec_home != None and os.path.isdir(spec_home):
    tag_spec = read_spec_files(spec_home, tag_spec, True)

  # Abort if spec yaml is invalid
  schema = yaml.safe_load(SCHEMA_TEXT)
  v = TfDockerTagValidator(schema, partials=partials)
  if not v.validate(tag_spec):
    eprint('> Error: Combined spec is invalid! The errors are:')
    eprint(yaml.dump(v.errors, indent=2))
    exit(1)
  tag_spec = v.normalized(tag_spec)
  return tag_spec

def merge_dir(dir_1, dir_2):
  merged_dir = '/tmp/partials'
  if os.path.exists(merged_dir):
    shutil.rmtree(merged_dir)
  if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)
  copy_tree(dir_1, merged_dir)
  copy_tree(dir_2, merged_dir)
  return merged_dir


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.generate_new_spec:
    auto_generate_model_spec(FLAGS.generate_new_spec.strip())

    # exit since the user needs to fill in things like partials and build args
    # we can't build dockerfiles/images in the same run as creating the spec
    sys.exit(0)

  # Get existing partial contents for centos or ubuntu
  common_partials = os.path.join(FLAGS.partial_dir, 'common')
  os_partials = os.path.join(FLAGS.partial_dir, 'ubuntu')
  if any("centos" in arg or "spr" in arg for arg in FLAGS.arg):
    os_partials = os.path.join(FLAGS.partial_dir, 'centos')

  partials_dir = merge_dir(os_partials, common_partials)
  partials = gather_existing_partials(partials_dir)

  # read in all spec files
  tag_spec = get_tag_spec(FLAGS.spec_dir, partials)

  # characters for underlining headers
  underlined = '\033[4m'
  end_underline = '\033[0m'

  # Assemble tags and images used to build them
  all_tags = assemble_tags(tag_spec, FLAGS.arg, FLAGS.release, partials)

  if FLAGS.print_models:
    models=["all"]
    for tag, tag_defs in all_tags.items():
      for tag_def in tag_defs:
        if 'is_dockerfiles' in tag_def:
          if tag_def['is_dockerfiles'] == True:
            lst = re.findall('{([^{}]*)}',tag_def['tag_spec'])
            if lst is not None and len(lst) > 0:
              target = lst[len(lst)-1]
              models.append(target)
    eprint('{}'.format(" ".join(models)))
    sys.exit(0)

  if FLAGS.list_images:
    for tag, tag_defs in all_tags.items():
      for tag_def in tag_defs:
        if 'tag_spec' in tag_def and 'is_dockerfiles' in tag_def:
          if tag_def['is_dockerfiles'] == True:
            lst = re.findall('{([^{}]*)}',tag_def['tag_spec'])
            if lst is not None and len(lst) > 0:
              target = lst[len(lst)-1]
              image = '{}:{}'.format(FLAGS.repository, tag)
              eprint('{} {} {}'.format(tag_def['release'], target,image))
    sys.exit(0)

  if FLAGS.list_packages:
    for tag, tag_defs in all_tags.items():
      for tag_def in tag_defs:
        if 'tag_spec' in tag_def:
          lst = re.findall('{([^{}]*)}',tag_def['tag_spec'])
          if lst is not None and len(lst) > 0:
            target = lst[len(lst)-1]
            package = get_package_name(tag_def)
            if package != None:
              tar_file = os.path.join(FLAGS.output_dir, "{}.tar.gz".format(package))
              eprint('{} {}'.format(target, tar_file))
    sys.exit(0)

  # Empty Dockerfile directory if building new Dockerfiles
  if FLAGS.construct_dockerfiles and not FLAGS.only_tags_matching:
    delete_dockerfiles(all_tags, FLAGS.dockerfile_dir)

  # Set up Docker helper
  dock = docker.from_env()

  # Login to Docker if uploading images
  if FLAGS.upload_to_hub:
    if not FLAGS.hub_username:
      print('> Error: please set --hub_username when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_repository:
      eprint(
          '> Error: please set --hub_repository when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_password:
      eprint('> Error: please set --hub_password when uploading to Dockerhub.')
      exit(1)
    dock.login(
        username=FLAGS.hub_username,
        password=FLAGS.hub_password,
    )

  # Each tag has a name ('tag') and a definition consisting of the contents
  # of its Dockerfile, its build arg list, etc.
  failed_tags = []
  succeeded_tags = []
  failed_dockerfiles = []
  succeeded_dockerfiles = []
  failed_docs = []
  succeeded_docs = []
  failed_packages = []
  succeeded_packages = []

  for tag, tag_defs in all_tags.items():
    for tag_def in tag_defs:
      eprint('> Working on {}'.format(tag), verbose=FLAGS.verbose)

      if FLAGS.exclude_tags_matching and re.match(FLAGS.exclude_tags_matching,
                                                  tag):
        eprint('>> Excluded due to match against "{}".'.format(
            FLAGS.exclude_tags_matching), verbose=FLAGS.verbose)
        continue

      if FLAGS.only_tags_matching and not [x for x in FLAGS.only_tags_matching if re.match(x, tag)]:
          eprint('>> Excluded due to failure to match against "{}".'.format(
              FLAGS.only_tags_matching), verbose=FLAGS.verbose)
          continue

      # Write model packages to the output_dir
      if FLAGS.build_packages:
        write_package(tag_def, succeeded_packages, failed_packages)

      # Write releases marked "is_dockerfiles" into the Dockerfile directory
      if FLAGS.construct_dockerfiles and tag_def['is_dockerfiles']:
        path = os.path.join(FLAGS.dockerfile_dir,
                            tag_def['dockerfile_subdirectory'],
                            tag_def['dockerfile_tag_name'] + '.Dockerfile')
        eprint('>> Writing {}...'.format(path), verbose=FLAGS.verbose)
        if not FLAGS.dry_run:
          mkdir_p(os.path.dirname(path))
          dockerfile_contents = tag_def['dockerfile_contents']

          # Go through our dict of build args and fill in the values in the dockerfile
          for build_arg_name, build_arg_value in tag_def['cli_args'].items():
            search_string = 'ARG {}[\S]*'.format(build_arg_name)
            replace_string = 'ARG {}="{}"'.format(build_arg_name, build_arg_value)
            dockerfile_contents = re.sub(search_string, replace_string, dockerfile_contents)

          with open(path, 'w', encoding="utf-8") as f:
            f.write(dockerfile_contents)

          if os.path.exists(path):
              succeeded_dockerfiles.append(os.path.relpath(path, FLAGS.dockerfile_dir))
          else:
              failed_dockerfiles.append(os.path.relpath(path, FLAGS.dockerfile_dir))

      if FLAGS.generate_deployments:
        eprint('> Generating deployment for {}'.format(tag), verbose=FLAGS.verbose)
        write_deployment(tag_def)

      if FLAGS.generate_deployments_tests:
        eprint('> Generating tests for K8s packages and deployments {}'.format(tag), verbose=FLAGS.verbose)
        model_dir = os.path.join(os.getcwd(), "models")
        tests_bats = os.path.join(model_dir, 'tools', 'tests', 'model-builder.bats')
        model_name = get_package_name(tag_def)
        model_deployments_dir = os.path.join(os.getcwd(), FLAGS.deployment_dir, model_name)
        try:
          eprint('>> Writing {}...'.format(tests_bats), verbose=FLAGS.verbose)
          with open(tests_bats, 'w', encoding="utf-8") as f:
            # add at the beginning of the test
            f.write('#!/usr/bin/env tools/tests/bin/bats\npushd {}\n'.format(model_dir))
            # get test uri from the model spec file and append it to model-builder.bats file
            for test in tag_def['runtime']['tests']:
              if not 'uri' in test:
                eprint("uri missing in tests values for {}".format(model_name))
                continue
              test_uri = os.path.join(model_dir, test['uri'])
              f.write(set_test_args(tag_def, test, test_uri))
              f.write('\n')
        except Exception as e:
          eprint("Error while writing tests for {}".format(tag))
          eprint(e)

      if FLAGS.generate_documentation:
        documentation_list = tag_def['documentation']
        doc_paths = [os.path.join(doc['uri'], doc['name']) for doc in documentation_list]
        if len(doc_paths) != len(set(doc_paths)):
            eprint('ERROR: The documentation for {} has more than one item with the same uri/name '
                   'path, which means that one doc would overwrite the other.'.format(tag))
            eprint('\n'.join(doc_paths))
            sys.exit(1)

        for documentation in documentation_list:
          text_replace = documentation['text_replace'] \
            if 'text_replace' in documentation else {}
          if 'contents' in documentation:
            readme = os.path.join(documentation['uri'], documentation['name'])
            eprint('>> Writing {}...'.format(readme), verbose=FLAGS.verbose)
            try:
              with open(readme, 'w', encoding="utf-8") as f:
                for k, v in text_replace.items():
                  documentation['contents'] = documentation['contents'].replace(k, v)
                f.write(documentation['contents'])
              succeeded_docs.append(readme)
            except Exception as e:
              eprint("Error while writing documentation file ({}) for {}".format(readme, tag))
              eprint(e)
              failed_docs.append(readme)

      # Don't build any images for dockerfile-only releases
      if not FLAGS.build_images:
        continue

      # Only build images for host architecture
      proc_arch = platform.processor()
      is_x86 = proc_arch.startswith('x86')
      if (is_x86 and any([arch in tag for arch in ['ppc64le']]) or
          not is_x86 and proc_arch not in tag):
        continue

      # Generate a temporary Dockerfile to use to build, since docker-py
      # needs a filepath relative to the build context (i.e. the current
      # directory)
      dockerfile = os.path.join(FLAGS.dockerfile_dir, tag + '.temp.Dockerfile')
      if not FLAGS.dry_run:
        with open(dockerfile, 'w', encoding="utf-8") as f:
          f.write(tag_def['dockerfile_contents'])
      eprint('>> (Temporary) writing {}...'.format(dockerfile), verbose=FLAGS.verbose)

      repo_tag = '{}:{}'.format(FLAGS.repository, tag)
      eprint('>> Building {} using build args:'.format(repo_tag), verbose=FLAGS.verbose)
      for arg, value in tag_def['cli_args'].items():
        eprint('>>> {}={}'.format(arg, value), verbose=FLAGS.verbose)

      # Note that we are NOT using cache_from, which appears to limit
      # available cache layers to those from explicitly specified layers. Many
      # of our layers are similar between local builds, so we want to use the
      # implied local build cache.
      tag_failed = False
      image, logs = None, []
      if not FLAGS.dry_run:
        try:
          quiet = True if FLAGS.verbose == False else False
          # Use low level APIClient in order to stream log output
          resp = dock.api.build(
              timeout=FLAGS.hub_timeout,
              path='.',
              nocache=FLAGS.nocache,
              quiet=quiet,
              dockerfile=dockerfile,
              buildargs=tag_def['cli_args'],
              tag=repo_tag)
          last_event = None
          image_id = None
          # Manually process log output extracting build success and image id
          # in order to get built image
          while True:
            try:
              output = next(resp).decode('utf-8')
              json_output = json.loads(output.strip('\r\n'))
              if 'stream' in json_output:
                eprint(json_output['stream'], end='', verbose=FLAGS.verbose)
                match = re.search(r'(^Successfully built |sha256:)([0-9a-f]+)$',
                                  json_output['stream'])
                if match:
                  image_id = match.group(2)
                last_event = json_output['stream']
                # collect all log lines into the logs object
                logs.append(json_output)
            except StopIteration:
              eprint(('Docker image build complete.'), verbose=FLAGS.verbose)
              break
            except ValueError:
              eprint('> Error parsing from docker image build: {}'.format(output))
          # If Image ID is not set, the image failed to built properly. Raise
          # an error in this case with the last log line and all logs
          if image_id:
            image = dock.images.get(image_id)
          else:
            raise docker.errors.BuildError(last_event or 'Unknown', logs)

          # Run tests if requested, and dump output
          # Could be improved by backgrounding, but would need better
          # multiprocessing support to track failures properly.
          if FLAGS.run_tests_path:
            if not tag_def['tests']:
              eprint(('>>> No tests to run.'), verbose=FLAGS.verbose)
            for test in tag_def['tests']:
              eprint(('>> Testing {}...'.format(test)), verbose=FLAGS.verbose)
              container, = dock.containers.run(
                  image,
                  '/tests/' + test,
                  working_dir='/',
                  log_config={'type': 'journald'},
                  detach=True,
                  stderr=True,
                  stdout=True,
                  volumes={
                      FLAGS.run_tests_path: {
                          'bind': '/tests',
                          'mode': 'ro'
                      }
                  },
                  runtime=tag_def['test_runtime']),
              ret = container.wait()
              code = ret['StatusCode']
              out = container.logs(stdout=True, stderr=False)
              err = container.logs(stdout=False, stderr=True)
              container.remove()
              if out:
                eprint('>>> Output stdout:', verbose=FLAGS.verbose)
                eprint(out.decode('utf-8'), verbose=FLAGS.verbose)
              else:
                eprint('>>> No test standard out.', verbose=FLAGS.verbose)
              if err:
                eprint('>>> Output stderr:')
                eprint(err.decode('utf-8'))
              else:
                eprint('>>> No test standard err.')
              if code != 0:
                eprint('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                failed_tags.append(repo_tag)
                tag_failed = True
                if FLAGS.stop_on_failure:
                  eprint('>> ABORTING due to --stop_on_failure!')
                  exit(1)
              else:
                eprint('>> Tests look good!', verbose=FLAGS.verbose)

        except docker.errors.BuildError as e:
          eprint('>> {} failed to build with message: "{}"'.format(
              repo_tag, e.msg))
          eprint(json_output)
          eprint('>> Build logs follow:')
          log_lines = [l.get('stream', '') for l in e.build_log]
          eprint(''.join(log_lines))
          failed_tags.append(repo_tag)
          tag_failed = True
          if FLAGS.stop_on_failure:
            eprint('>> ABORTING due to --stop_on_failure!')
            exit(1)

        # Clean temporary dockerfiles if they were created earlier
        if not FLAGS.keep_temp_dockerfiles:
          os.remove(dockerfile)

      # Upload new images to DockerHub as long as they built + passed tests
      if FLAGS.upload_to_hub:
        if not tag_def['upload_images']:
          continue
        if tag_failed:
          continue

        eprint('>> Uploading to {}:{}'.format(FLAGS.hub_repository, tag), verbose=FLAGS.verbose)
        if not FLAGS.dry_run:
          p = multiprocessing.Process(
              target=upload_in_background,
              args=(FLAGS.hub_repository, dock, image, tag))
          p.start()

      if not tag_failed:
        succeeded_tags.append(repo_tag)

  if failed_tags:
    eprint(
        '> Some tags failed to build or failed testing, check scrollback for '
        'errors: {}'.format(','.join(failed_tags)))
    exit(1)

  release_group_str = ""
  if FLAGS.release:
      release_group_str = "(release group: {})".format(", ".join(FLAGS.release))

  # Print failed/succeeded dockerfiles
  if failed_dockerfiles:
    failed_dockerfiles.sort()
    eprint('Some dockerfiles failed to be written {}. Scroll up to check for '
           'errors: {}'.format(release_group_str, ','.join(failed_dockerfiles)))

  if succeeded_dockerfiles:
    succeeded_dockerfiles.sort()
    eprint('{}Dockerfiles written {}:{}'.format(underlined, release_group_str, end_underline))
    eprint("\n".join(succeeded_dockerfiles))

  # Print failed/succeeded documentation files
  if failed_docs:
    failed_docs.sort()
    eprint('Some documentation failed to be written {}. Scroll up to check for '
           'errors. Docs failed for: {}'.format(release_group_str, ', '.join(failed_docs)))

  if succeeded_docs:
    succeeded_docs.sort()
    eprint('{}Documentation files written {}:{}'.format(underlined, release_group_str,
                                                  end_underline))
    eprint("\n".join(succeeded_docs))

  # Print failed/succeeded packages
  if failed_packages:
    failed_packages.sort()
    eprint('Some packages failed to be written {}. Scroll up to check for '
           'errors. Docs failed for: {}'.format(release_group_str, ', '.join(failed_packages)))

  if succeeded_packages:
    succeeded_packages.sort()
    eprint('{}Packages written {}:{}'.format(underlined, release_group_str, end_underline))
    eprint("\n".join(succeeded_packages))

  eprint('> Writing built{} tags to standard out.'.format(
      ' and tested' if FLAGS.run_tests_path else ''), verbose=FLAGS.verbose)
  images_built_var = os.environ.get("IMAGES_BUILT", "")
  if not FLAGS.dry_run:
    for tag in succeeded_tags:
      eprint('{}:{}'.format(FLAGS.repository, tag), verbose=FLAGS.verbose)
      images_built_var = tag if not images_built_var else "{},{}".format(images_built_var, tag)

    # Update the env var of images built
    if FLAGS.build_images and images_built_var:
      print("IMAGES_BUILT={}".format(images_built_var))


if __name__ == '__main__':
  app.run(main)
