#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021-2023 Intel Corporation
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

#
import sys
import configparser
import json
import os
import fnmatch
import psutil
import yaml

from importlib import util
tensorflow_found = util.find_spec("tensorflow") is not None
pytorch_found = util.find_spec("torch") is not None
pytorch_ext_found = util.find_spec("intel_pytorch_extension") is not None

try:
    import tensorflow as tf
    from tensorflow.python.client import timeline
    from tensorflow.python.training import training_util
except ImportError as e:
    print(e)
    print("can't import tensorflow module")
    pass

try:
    from git import Repo
    has_git = True
except ImportError as e:
    print(e)
    print("can't import git module")
    has_git = False
    pass

class TensorflowUtils:

    def is_mkl_enabled(self):
        mkl_enabled = False
        major_version = int(tf.__version__.split(".")[0])
        minor_version = int(tf.__version__.split(".")[1])
        if major_version >= 2:
            if minor_version < 5:
                from tensorflow.python import _pywrap_util_port
            else:
                from tensorflow.python.util import _pywrap_util_port
                onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
            mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
        else:
            mkl_enabled = tf.pywrap_tensorflow.IsMklEnabled()
        return mkl_enabled


class GitOps:

    def __init__(self, repopath='./'):
        self.repopath = repopath
        if has_git is True:
            try:
                self.repo = Repo(repopath)
            except:
                print('no repo, init one')
                self.repo = Repo.init(repopath)
                self.repo.git.add(A=True)
                self.repo.git.commit(m='add new files')
                pass
        return

    def check_git_status(self):
        status = self.repo.git.status()
        if status.find("middle of an am session") != -1 or self.repo.is_dirty():
            print("middle of an am session or repo is dirty")
            return False
        return True

    def find_commit_with_keyword(self, keyword, search_commits_depth=5):
        ret = False
        if has_git is False:
            return ret
        try:
            repo = Repo(self.repopath)
        except:
            print('no repo, init one')
            repo = Repo.init(self.repopath)
            repo.git.add(A=True)
            repo.git.commit(m='add new files')
            pass

        if not repo.bare:
            try:
                commits = list(repo.iter_commits())[:search_commits_depth]
                for commit in commits:
                    if commit.summary.find(keyword) != -1:
                        ret = True
                        return ret
            except Exception:
                print("EXCEPTION : Find commit %s ", keyword)
                pass
        return ret


class PlatformUtils:

    def __init__(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # Import platform_util and print CPU information
        if 'AIReferenceRoot' in os.environ and os.environ['AIReferenceRoot']:
            repo_root = os.environ['AIReferenceRoot']
            platform_util = os.path.join(repo_root, 'benchmarks/common')
        else:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            platform_util = os.path.join(file_dir, '../../benchmarks/common')
        sys.path.insert(0, platform_util)
        import platform_util
        cpu_info = platform_util.CPUInfo()
        print("=" * 20, "CPU Info", "=" * 20)
        # number of cores
        print("Physical cores per socket:", cpu_info.cores_per_socket)
        print("Total physical cores:", cpu_info.cores)
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print("Max Frequency:", cpufreq.max)
        print("Min Frequency:", cpufreq.min)
        print("Socket Number:", cpu_info.sockets)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        self.cpu_socket_count = cpu_info.sockets
        self.svmem = svmem


class CommonUtils:

    def __init__(self):
        return

    def found_files_in_folder(self, pattern, path):
        listOfFiles = os.listdir(path)
        foundfiles = []
        founpaths = []
        for f in listOfFiles:
            if fnmatch.fnmatch(f, pattern):
                foundfiles.append(f)
                founpaths.append(path + os.sep + f)
        return foundfiles, founpaths

    def found_files_in_folders(self, pattern, paths):
        foundfiles = []
        foundpaths = []
        for path in paths:
            files, paths = self.found_files_in_folder(pattern, path)
            foundfiles += files
            foundpaths += paths
        return foundfiles, foundpaths

class AIReferenceConfigFile:

    def __init__(self, confpath=None):
        self.configpath = confpath
        self.wget = ''
        self.data_download = ''
        self.data_location = ''
        if tensorflow_found == True:
            self.tf_util = TensorflowUtils()
        self.model_name = ''
        self.script = ''
        self.mode = ''
        self.framework = ''
        self.device = ''
        self.precision = ''
        self.ai_type = ''
        self.custom_args = ''
        self.json_fname = ''
        self.json_fname = 'stock_'
        self.patches = ''
        self.patched = False
        self.patches_keyword = ''

    def read_section(self):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        return config.sections()

    def read_supported_section(self):
        supported_sections = []

        with open(self.configpath, 'r') as stream:
            config = yaml.safe_load(stream)

        for section in config:
            supported_sections.append(section)

        return supported_sections

    def convert_configs_to_pd_dataframe(self):

        config = configparser.ConfigParser()
        config.read(self.configpath)
        import pandas as pd

        index_list = []
        data_list = []
        columns_list = ['benchmark', 'model-name', 'mode', 'precision', 'device', 'framework', 'patches', 'json-fname']
        for section in config.sections():
            index_list.append(section)
            data = []
            data.append(section)
            data.append(config.get(section, columns_list[1]))
            data.append(config.get(section, columns_list[2]))
            data.append(config.get(section, columns_list[3]))
            data.append(config.get(section, columns_list[4]))
            data.append(config.get(section, columns_list[5]))
            data.append(config.get(section, columns_list[6]))
            data.append(config.get(section, columns_list[7]))
            data_list.append(data)
        df = pd.DataFrame(data_list, columns=columns_list)

        df_types = df.groupby([columns_list[1], columns_list[2]]).filter(lambda x: len(x) >= 2)

        df_types_obj = df_types.groupby([columns_list[1], columns_list[2]])

        return df, df_types, df_types_obj

    def read_value_from_section(self, model_name, key):
        with open(self.configpath, 'r') as stream:
            config = yaml.safe_load(stream)
        string_val = config.get(model_name, key)
        return string_val

    def write_value_from_section(self, model_name, key, val):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        config.set(model_name, key, val)

        # save to a file
        with open(self.configpath, 'w') as configfile:
            config.write(configfile)
        return

    def parsing_custom_args(self, model_name, custom_args):
        configvals = []
        if custom_args.count('$') >= 2:
            read_var = custom_args.split('$')[1]
            replace_var = self.read_value_from_section(model_name, read_var)
            custom_args = custom_args.replace('$' + read_var + '$', replace_var)
        custom_args_list = custom_args.split(" ")
        for arg in custom_args_list:
            if arg != '':
                configvals.append(arg)
        return configvals

    def read_config(self, model_name):
        configs = {}

        with open(self.configpath, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

            for entry in data:
                if entry['name'] == model_name:
                    model_data = entry
                    break
                else:
                    model_data = None

            if model_data:
                for key, value in model_data.items():
                    if key == 'name':
                        configs['name'] = value
                    elif key == 'ai-type':
                        configs['ai-type'] = value
                    elif key == 'model-name':
                        configs['model-name'] = value
                    elif key == 'mode':
                        configs['mode'] = value
                    elif key == 'framework':
                        configs['framework'] = value
                    elif key == 'device':
                        configs['device'] = value
                    elif key == 'data-download':
                        configs['data-download'] = value
                    elif key == model_name.split()[0]:
                        for sub_entry in value:
                            precision = sub_entry['precision']
                            scripts = sub_entry['script']
                            wget = sub_entry['wget']
                            if model_name.split()[0] not in configs:
                                configs[model_name.split()[0]] = []
                            configs[model_name.split()[0]].append({'precision': precision, 'script': scripts, 'wget': wget})

        return configs

    def get_parameters(
            self, model_name, configvals):
        benchmark_argvs = []
        benchmark_argvs = benchmark_argvs + configvals
        return benchmark_argvs

    def find_pretrained_model_in_folder(self, uncompress_folderpath, pattern='*.pb'):
        utils = CommonUtils()
        if uncompress_folderpath.find(pattern[1:]) != -1:
            return uncompress_folderpath
        pb_filename = ''
        pb_files, pb_paths = utils.found_files_in_folder(pattern, uncompress_folderpath)
        if len(pb_files) != 0:
            pb_filename = os.sep + pb_files[0]
        pretrain_model_path = uncompress_folderpath + pb_filename
        return pretrain_model_path

    def untar_file(self, filepath, extracted_fd='./'):
        import tarfile
        extractpath = ''
        tar = tarfile.open(filepath)
        for tarinfo in tar:
            if tarinfo.isdir():
                print(tarinfo.name)
                extractpath = tarinfo.name
                break
        target_path = extracted_fd + os.sep + extractpath
        if os.path.exists(target_path) is False:
            tar.extractall(path=extracted_fd)
        tar.close()
        return extractpath

    def uncompress_file(self, filepath, pretrainfd='pretrained', current_path='./'):
        import shutil
        uncompress_path = filepath
        full_filename = filepath.split(os.sep)[-1]

        file_ext = full_filename.split('.')[-1]
        filename = full_filename.split('.')[0]
        cmd = ''
        if file_ext == 'zip':
            cmd = "unzip " + filepath
        elif file_ext == 'gz':
            uncompress_foldername = self.untar_file(filepath, extracted_fd=pretrainfd)
            uncompress_path = filepath.replace(full_filename, uncompress_foldername)
        if cmd != '':
            os.system(cmd)
            if os.path.exists(pretrainfd + os.sep + filename) is False:
                shutil.move(filename, pretrainfd)

            uncompress_path = os.path.splitext(filepath)[0]

        return uncompress_path

    def download_dataset(self, datasetfd='dataset', current_path='./'):
        cmd = self.data_download
        dataset_path = current_path + os.sep + datasetfd
        if os.path.exists(dataset_path) is True:
            return dataset_path
        os.system(cmd)
        print('Downloaded the model in:', dataset_path)
        return dataset_path

    def download_pretrained_model(self, wget='', pretrainfd='pretrained', current_path='./'):
        import shutil
        cmd = "wget " + wget + ' -P ' + current_path + os.sep + pretrainfd
        filename = wget.split('/')[-1]
        pretrain_model_path = current_path + os.sep + pretrainfd + os.sep + filename
        if os.path.exists(pretrain_model_path) is True:
            return pretrain_model_path
        os.system(cmd)
        if os.path.exists(pretrainfd) is False:
            os.mkdir(pretrainfd)
        if os.path.exists(pretrainfd + os.sep + filename) is False:
            shutil.move(filename, pretrainfd)
        print('Downloaded the model in:', pretrain_model_path)
        return pretrain_model_path
