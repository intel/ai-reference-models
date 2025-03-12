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
import os
import fnmatch
import psutil
import yaml
from prettytable import PrettyTable
from IPython.display import display
import ipywidgets as widgets

from importlib import util
tensorflow_found = util.find_spec("tensorflow") is not None
pytorch_found = util.find_spec("torch") is not None
pytorch_ext_found = util.find_spec("intel_pytorch_extension") is not None

try:
    from git import Repo
    has_git = True
except ImportError as e:
    print(e)
    print("can't import git module")
    has_git = False
    pass


class GitOps:

    def __init__(self, repopath='./'):
        self.repopath = repopath
        if has_git is True:
            try:
                self.repo = Repo(repopath)
            except Exception:
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
        except Exception:
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


class GeneralConfigFile:
    def __init__(self, ai_root):
        self.root = ai_root
        self.framework = ''
        self.device = ''
        # Empty for CPU
        self.device_series = ''
        # only move on to the the proceeding AIReference setup when True
        self.success = False

        # widgets
        self.hardware_dropdown = widgets.Dropdown(
            options=['CPU', 'GPU'],
            value=None,
            description='Choose Hardware:',
            disabled=False,
        )

        self.framework_dropdown = widgets.Dropdown(
            options=['PyTorch', 'TensorFlow'],
            value=None,
            description='Choose Framework:',
            disabled=False,
        )

        self.gpu_series_dropdown = widgets.Dropdown(
            options=['Flex', 'Max', 'Arc'],
            value=None,
            description='Choose GPU Series:',
            disabled=False,
        )

        self.intel_oneapi_dropdown = widgets.Dropdown(
            options=['Yes', 'No'],
            value=None,
            description='Intel® oneAPI Base Toolkit installed?:',
            disabled=False,
        )

        self.oneapi_path_input = widgets.Text(
            value='',
            placeholder='Input oneAPI installation path',
            description='oneAPI Path:',
            disabled=False,
        )

    def toggle_events(self):
        def on_hardware_change(change):
            if change.new == 'GPU':
                display(self.gpu_series_dropdown)
            else:
                self.gpu_series_dropdown.close()
                self.intel_oneapi_dropdown.close()
                self.oneapi_path_input.close()

        def on_framework_change(change):
            if self.hardware_dropdown.value == 'GPU':
                if change.new == 'PyTorch':
                    display(self.intel_oneapi_dropdown)
                else:
                    self.intel_oneapi_dropdown.close()
                    self.oneapi_path_input.close()
                update_gpu_series_options()
            else:
                self.intel_oneapi_dropdown.close()
                self.oneapi_path_input.close()

        def update_gpu_series_options():
            if (self.hardware_dropdown.value != 'GPU'):
                raise Exception("Debugging: Did not selected GPU but proceeded anyways")
            selected_framework = self.framework_dropdown.value
            if selected_framework == 'PyTorch':
                self.gpu_series_dropdown.options = ['Flex', 'Max', 'Arc']
            elif selected_framework == 'TensorFlow':
                self.gpu_series_dropdown.options = ['Flex', 'Max']
            else:
                self.gpu_series_dropdown.options = ['Please choose the framework']

        def on_intel_oneapi_change(change):
            if change.new == 'Yes':
                display(self.oneapi_path_input)
                print("Default path for oneAPI Base Toolkit is: /opt/intel/oneapi")
            else:
                self.oneapi_path_input.close()
                if change.new == 'No':
                    # Intel® oneAPI Base Toolkit is not installed
                    print("Intel® oneAPI Base Toolkit is not installed.")
                    print(
                        "Follow instructions at [Intel® oneAPI Base Toolkit Download page]"
                        "(https://www.intel.com/content/www/us/en/developer/tools/oneapi"
                        "/base-toolkit-download.html?operatingsystem=linux) "
                        "to setup the package manager repository."
                    )
                    print("Once Intel® oneAPI Base Toolkit is installed on the machine, please re-run this cell")
                    return

        self.hardware_dropdown.observe(on_hardware_change, names='value')
        self.framework_dropdown.observe(on_framework_change, names='value')
        self.intel_oneapi_dropdown.observe(on_intel_oneapi_change, names='value')

        if (self.hardware_dropdown.value and self.framework_dropdown.value and
                (self.hardware_dropdown.value == 'CPU' or
                    (self.hardware_dropdown.value == 'GPU' and self.gpu_series_dropdown.value))):
            self.success = True
        display(self.hardware_dropdown)
        display(self.framework_dropdown)


class AIReferenceConfigFile:

    def __init__(self, confpath, AIpath):
        self.configpath = confpath
        self.wget = ''
        self.data_download = ''
        self.data_location = ''
        self.model_name = ''
        self.script = ''
        self.mode = ''
        self.framework = ''
        self.device = ''
        self.precision = ''
        self.test_mode = ''
        # self.ai_type = ''
        self.custom_args = ''
        self.json_fname = ''
        self.json_fname = 'stock_'
        self.patches = ''
        self.patched = False
        self.patches_keyword = ''

        # paths can be stored in the config object.
        self.ai_root = AIpath
        self.notebook_root = self.ai_root + os.sep + 'notebooks'
        self.profile_root = self.notebook_root + os.sep + 'profiling'
        self.exports = []
        self.additional_commands = []

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
                    # elif key == 'ai-type':
                    #    configs['ai-type'] = value
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
                    elif key == 'precision':
                        configs['precision'] = value
                    elif key == 'test_mode':
                        configs['test_mode'] = value
                    elif key == 'wget':
                        configs['wget'] = value
                    elif key == model_name.split()[0]:
                        for sub_entry in value:
                            precision = sub_entry['precision']
                            test_mode = sub_entry['test_mode']
                            scripts = sub_entry['script']
                            wget = sub_entry['wget']
                            if model_name.split()[0] not in configs:
                                configs[model_name.split()[0]] = []
                            configs[model_name.split()[0]].append({'precision': precision,
                                                                   'test_mode': test_mode,
                                                                   'script': scripts,
                                                                   'wget': wget})

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
        print(pretrain_model_path)
        if os.path.exists(pretrain_model_path) is True:
            return pretrain_model_path
        os.system(cmd)
        if os.path.exists(pretrainfd) is False:
            os.mkdir(pretrainfd)
        if os.path.exists(pretrainfd + os.sep + filename) is False:
            print(pretrainfd + os.sep + filename)
            shutil.move(filename, pretrainfd)
        print('Downloaded the model in:', pretrain_model_path)
        return pretrain_model_path

    def model_selection(self):

        sections = self.read_supported_section()
        models_table = PrettyTable(["Index", "Model Name", "Framework", "Mode", "Device"])

        for index, section in enumerate(sections):
            model_name = section.get('model-name', 'Unknown')
            # ai_type = section.get('ai-type', 'Unknown')
            framework = section.get('framework', 'Unknown')
            mode = section.get('mode', 'Unknown')
            device = section.get('device', 'Unknown')
            models_table.add_row([index, model_name, framework, mode, device])

        print("Supported Models: ")
        display(models_table)

        model_index = int(input('Input an index number of a model: ')) \
            if 'MODEL_1_INDEX' not in os.environ else int(os.environ['MODEL_1_INDEX'])

        if not 0 <= model_index < len(sections):
            raise Exception("Invalid choice for model index")

        print('Selected: ', sections[model_index]['name'])
        return sections[model_index]

    def device_specific(self, model_section, device, yaml_file, framework):
        """
        Handles model-specific configuration by parsing YAML files with different structures.
        """
        # Set initial properties
        self.model_name = model_section['name']
        self.framework = model_section['framework']
        self.mode = model_section['mode'][int(input('0 for training and 1 for inference: '))] \
            if len(model_section['mode']) > 1 else model_section['mode'][0]

        # Load model-specific configuration
        model_specific_config = AIReferenceConfigFile(yaml_file, self.ai_root)
        model_specific_section = model_specific_config.read_supported_section()[0]

        # Determine the precision options key dynamically
        # Use 'model-name' if available, else fallback
        precision_key = model_section.get('model-name', self.model_name.split()[0])

        # Get precision options
        model_precisions = model_specific_section.get(precision_key)

        if not model_precisions:
            raise ValueError(f"No precision options found for model '{self.model_name}'.")

        # Display precision options
        model_precision_table = PrettyTable(["Index", "Precision"])
        for index, precision in enumerate(model_precisions):
            model_precision_table.add_row([index, precision['precision']])
        display(model_precision_table)

        # Select precision
        precision_index = int(input('Select an index number for the precision: '))
        if not 0 <= precision_index < len(model_precisions):
            raise ValueError("Invalid index for precision.")

        # Get selected precision details
        model_precision_section = model_precisions[precision_index]
        self.precision = model_precision_section['precision']
        self.wget = model_precision_section.get('wget', '')

        if framework == 'TensorFlow' and device == 'CPU':
            # TensorFlow-specific handling
            model_precision_script = model_precision_section.get('script', [])
            if not model_precision_script:
                raise ValueError(f"No scripts found for model '{self.model_name}' with precision '{self.precision}'.")
            print(f"Available Scripts for {self.precision}:")
            model_script_table = PrettyTable(["Index", "Script"])
            for index, script in enumerate(model_precision_script):
                model_script_table.add_row([index, script])
            display(model_script_table)

            # Select script
            model_precision_script_index = int(input('Input an index for the available script: '))
            if not 0 <= model_precision_script_index < len(model_precision_script):
                raise ValueError("Invalid index for the selected script.")
            self.script = model_precision_script[model_precision_script_index]
        else:
            # Non-TensorFlow-specific handling
            model_precision_test_mode = model_precision_section.get('test_mode', [])
            if not model_precision_test_mode:
                raise ValueError(
                    f"No test modes found for model '{self.model_name}' with precision '{self.precision}'."
                )
            print(f"Available Test Modes for {self.precision}:")
            model_test_mode_table = PrettyTable(["Index", "Test Mode"])
            for index, test_mode in enumerate(model_precision_test_mode):
                model_test_mode_table.add_row([index, test_mode])
            display(model_test_mode_table)

            # Select test mode
            model_precision_test_mode_index = int(input('Input an index for the available test mode: '))
            if not 0 <= model_precision_test_mode_index < len(model_precision_test_mode):
                raise ValueError("Invalid index for the selected test mode.")
            self.test_mode = model_precision_test_mode[model_precision_test_mode_index]

        # Print selected configuration
        print(f"Selected {self.model_name} {self.framework} {self.precision} "
              "{self.script if framework == 'TensorFlow' else self.test_mode}")

        return model_specific_section
