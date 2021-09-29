#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

try:
    import torch
    has_torch = True
except ImportError as e:
    print(e)
    print("WARNING: PyTorch not available")
    has_torch = False


try:
    import intel_pytorch_extension as ipex
    has_ipex = True
except ImportError as e:
    print(e)
    print("IPEX not available")
    has_ipex = False

class MKLUtils:
    def is_mkl_enabled(self):
        if has_torch:
            search_str="USE_MKL="
            offset = len(search_str)
            config_str = torch.__config__.show()
            idx = config_str.find(search_str)
            return config_str[idx+offset:idx+offset+2] == "ON"
        return None



import json
import os, fnmatch
import psutil
import subprocess

try:
    from git import Repo
    has_git = True
except ImportError as e:
    print(e)
    print("can't import git module")
    has_git = False
    pass

class PlatformUtils:

    def __init_(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # let's print CPU information
        file_dir = os.path.dirname(os.path.abspath(__file__))
        platform_util_path = os.path.join(file_dir, '../../../../../benchmarks/common')
        print(platform_util_path)
        sys.path.insert(0, platform_util_path)
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



class ConfigFile:

    def __init__(self, confpath='profiling/topo.ini'):
        self.configpath = confpath
        self.data_download = ''
        self.data_location = ''
        self.checkpoint = ''
        self.mkl_util = MKLUtils()
        self.json_fname = ''
        if self.mkl_util.is_mkl_enabled() is True:
            self.json_fname = 'mkl_'
        else:
            self.json_fname = 'stock_'
        if has_ipex:
            self.json_fname += 'ipex_'
        self.throughput_keyword = ''
        self.throughput_index = -1
        self.support_accuracy = False

    def read_section(self):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        return config.sections()

    def read_supported_section(self, accuracy_only=False):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        supported_sections = []
        for each_section in config.sections():
            is_supported = True
            for each_key, each_val in config.items(each_section):
                if each_key == 'support-accuracy':
                    if each_val is not None:
                        if eval(each_val) is False and accuracy_only is True:
                            is_supported = False
            if is_supported is True:
                supported_sections.append(each_section)

        return supported_sections

    def convert_configs_to_pd_dataframe(self):

        config = configparser.ConfigParser()
        config.read(self.configpath)
        import pandas as pd
        import numpy as np

        index_list = []
        data_list = []
        columns_list = ['benchmark','model-name', 'mode', 'precision','patches','json-fname']
        for section in config.sections():
            index_list.append(section)
            data = []
            data.append(section)
            data.append(config.get(section, columns_list[1]))
            data.append(config.get(section, columns_list[2]))
            data.append(config.get(section, columns_list[3]))
            data.append(config.get(section, columns_list[4]))
            data.append(config.get(section, columns_list[5]))
            data_list.append(data)
        df = pd.DataFrame(data_list, columns = columns_list)

        df_types = df.groupby([ columns_list[1],  columns_list[2]]).filter(lambda x: len(x) >= 2)

        df_types_obj = df_types.groupby([ columns_list[1],  columns_list[2]])

        return df, df_types, df_types_obj

    def read_value_from_section(self, topo_name, key):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        string_val = config.get(topo_name, key)
        return string_val

    def write_value_from_section(self, topo_name, key, val):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        config.set(topo_name, key, val)

        # save to a file
        with open(self.configpath, 'w') as configfile:
             config.write(configfile)
        return

    def read_config(self, topo_name):
        configs = []
        config = configparser.ConfigParser()
        config.read(self.configpath)
        for each_section in config.sections():
            if each_section == topo_name:
                for each_key, each_val in config.items(each_section):
                    key = '--' + each_key
                    #if each_key == 'data-location':
                    #    if each_val is not None:
                    #        self.benchmark_only = False
                    if each_key == 'throughput-keyword':
                        if each_val is not None:
                            self.throughput_keyword = each_val
                    elif each_key == 'throughput-index':
                        if each_val is not None:
                            self.throughput_index = each_val
                    elif each_key == 'data-download':
                        if each_val is not None:
                            self.data_download = each_val
                    elif each_key == 'data-location':
                        if each_val is not None:
                            if each_val != '':
                                print("data-location : ", each_val)
                                configs.append(key)
                                configs.append(each_val)
                                self.data_location = each_val
                    elif each_key == 'checkpoint':
                        if each_val != '':
                            configs.append(key)
                            configs.append(each_val)
                            self.checkpoint = each_val
                        self.in_graph = 'NA'
                    elif each_key == 'json-fname':
                        if each_val is not None:
                            self.json_fname = self.json_fname + each_val
                    else:
                        if len(each_val) != 0 :
                            if each_val[0] == '=':
                                configs.append(key+each_val)
                            else:
                                configs.append(key)
                                configs.append(each_val)

        return configs

    def get_main_parameters(self, topology_name = "resnet50", mode = "infer", batch_size=1,
            has_ipex = False, precision="fp32", warmup_iter=30, log_path = ""):
        main_args = []
        main_args.append('--batch-size')
        main_args.append(str(batch_size))
        main_args.append("--arch")
        main_args.append(topology_name)
        main_args.append("--dummy")
        main_args.append("--evaluate")
        main_args.append("--pretrained")
        if has_ipex:
            main_args.append("--ipex")

        main_args.append("--precision")
        main_args.append(precision)
        main_args.append("--jit")
        main_args.append("--warmup-iterations")
        main_args.append(str(warmup_iter))
        if log_path !="":
            main_args.append("--log-path="+log_path)

        return main_args 

    def get_launch_parameters(
            self, configvals, thread_number=1,
            socket_number=1, num_inter_threads=0, num_intra_threads=0, accuracy_only=False):
        benchmark_argvs = []
        benchmark_argvs = benchmark_argvs + configvals

        benchmark_argvs.append('--nnodes=' + str(num_inter_threads))
        benchmark_argvs.append('--node_rank=' + str(socket_number))
        benchmark_argvs.append('--nproc_per_node=' + str(num_intra_threads))
        benchmark_argvs.append('--use_env')

        return benchmark_argvs
   
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
            cmd = "tar -xzvf  " + filepath
        if cmd != '':
            os.system(cmd)
            if os.path.exists(pretrainfd + os.sep + filename) is False:
                shutil.move(filename, pretrainfd)
            uncompress_path = filepath.split('.')[0]

        return uncompress_path

    def download_dataset(self, datasetfd='dataset', current_path='./'):
        import shutil
        cmd = self.data_download
        filename = self.wget.split('/')[-1]
        dataset_path = current_path + os.sep + datasetfd
        if os.path.exists(dataset_path) is True:
            return dataset_path
        os.system(cmd)
        print('Downloaded the model in:', dataset_path)
        return dataset_path

    def download_pretrained_model(self, pretrainfd='pretrained', current_path='./'):
        import shutil
        cmd = "wget " + self.wget
        filename = self.wget.split('/')[-1]
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


class PerfPresenter:

    def __init__(self, showAbsNumber=False):
        self.showAbsNumber = showAbsNumber
        pass

    def autolabel(self, ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                '{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')

    def draw_perf_diag(self, topo_name, a_means, b_means, a_label, b_label):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [topo_name]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, a_means, width, label=a_label)
        rects2 = ax.bar(x + width / 2, b_means, width, label=b_label)
        ax.set_ylabel('Throughput')
        ax.set_title('stock Torch vs Intel Torch')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        self.autolabel(ax, rects1)
        self.autolabel(ax, rects2)
        fig.tight_layout()
        plt.show()

    def draw_perf_ratio_diag(self, topo_name, a_means, b_means, a_label, b_label):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [topo_name]
        a_ratio = 1
        b_ratio = float(b_means / a_means)
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, a_ratio, width, label=a_label)
        rects2 = ax.bar(x + width / 2, b_ratio, width, label=b_label)
        ax.set_ylabel('Throughput Speedup ')
        ax.set_title('stock Torch vs Intel Torch')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        self.autolabel(ax, rects1)
        self.autolabel(ax, rects2)
        fig.tight_layout()
        plt.show()

    def create_csv_logfile(self, Type, filename):
        import csv
        import os.path
        if Type == 'training':
            fnames = ['ipex', 'elapsed_time']
        else:
            fnames = ['ipex', 'elapsed_time', 'throughput', 'accuracy']
        if os.path.isfile(filename):
            print('file exists')
        else:
            f = open(filename, 'w')
            with f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()

    def get_diff_from_csv_filenames(self, x, y):
        x_split=x.split('_')
        y_split=y.split('_')
        if len(x_split) != len(y_split):
            print("ERROR! can't two files have different formats")
            return '',''
        for i in range(len(x_split)):
            if x_split[i] !=  y_split[i]:
                break
        return x_split[i], y_split[i]

    def log_infer_perfcsv(self, elapsed_time, throughput, accuracy ,filename, ipex_enabled):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['ipex', 'elapsed_time', 'throughput', 'accuracy']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'ipex': ipex_enabled,
                 'elapsed_time': elapsed_time,
                 'throughput': throughput,
                 'accuracy': accuracy})

    def log_train_perfcsv(self, elapsed_time, filename, ipex_enabled):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['ipex', 'elapsed_time']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'ipex': ipex_enabled,
                 'elapsed_time': elapsed_time})

    def read_number_from_csv(self, filepath, framename):
        import csv
        import statistics
        f = open(filepath, 'r')
        col1 = []
        col2 = []
        mean1 = 0
        mean2 = 0
        stdev1 = 0
        stdev2 = 0
        with f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['ipex'] == 'True':
                    col1.append(float(row[framename]))
                else:
                    col2.append(float(row[framename]))

        if len(col1) > 0:
            mean1 = statistics.mean(col1)
        if len(col2) > 0:
            mean2 = statistics.mean(col2)
        if len(col1) > 1:
            stdev1 = statistics.stdev(col1)
        if len(col2) > 1:
            stdev2 = statistics.stdev(col2)
        return (mean1, mean2, len(col1), len(col2), stdev1, stdev2)

    def plot_perf_graph(self, ylabel, xlabel, stock_number, intel_number, stock_stdev, intel_stdev):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [xlabel]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, stock_number, width, yerr=stock_stdev, label='stock Torch')
        rects2 = ax.bar(x + width / 2, intel_number, width, yerr=intel_stdev, label='intel Torch')
        ax.set_ylabel(ylabel)
        ax.set_title('stock Torch vs Intel Torch')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    '{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

    def plot_perf_graph_v2(self, ylabel, xlabel, means_list, stddev_list, filepath_list, label_list,title='stock Torch vs Intel Torch'):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [xlabel]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()

        rects_list = []
        for i in range(len(filepath_list)):
            number = means_list[i]
            stddev = stddev_list[i]
            filepath = filepath_list[i]
            label = label_list[i]
            rects = ax.bar(x - width / 2 + width*i, number, width, yerr=stddev, label=label )
            rects_list.append(rects)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    '{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

        for i in range(len(rects_list)):
            autolabel(rects_list[i])
        fig.tight_layout()
        plt.show()

    def draw_perf_diag_from_csv(self, filepath, framename, y_axis_name, title):
        if self.showAbsNumber is False:
            return
        stock_means = []
        intel_means = []
        stock_stdev = []
        intel_stdev = []
        mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
        stock_means.append(float(mean2))
        intel_means.append(float(mean1))
        stock_stdev.append(float(stdev2))
        intel_stdev.append(float(stdev1))
        self.plot_perf_graph(y_axis_name, title, stock_means, intel_means, stock_stdev, intel_stdev)

    def draw_perf_ratio_diag_from_csv(self, filepath, framename, y_axis_name, title):
        stock_ratio_means = [1]
        intel_ratio_means = []
        stock_stdev = []
        intel_stdev = []
        mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
        if mean1 is 0 or mean2 is 0:
            print("ERROR. Users must run the benchmark with both Stock Torch and Intel Torch\n")
            return
        intel_ratio_means.append(float(mean1 / mean2))
        stock_stdev.append(float(stdev2 / mean2))
        intel_stdev.append(float(stdev1 / mean1))
        self.plot_perf_graph(y_axis_name, title, stock_ratio_means, intel_ratio_means, stock_stdev, intel_stdev)

    def draw_perf_diag_from_csvs(self, filepath_list,label_list ,framename, y_axis_name, x_axis_name, title, analyze_mkl=True):
        if self.showAbsNumber is False:
            return
        means_list = []
        stddev_list = []
        for filepath in filepath_list:
            means = []
            stdev = []
            mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
            if analyze_mkl == True:
                means.append(float(mean1))
                stdev.append(float(stdev1))
            else:
                means.append(float(mean2))
                stdev.append(float(stdev2))
            means_list.append(means)
            stddev_list.append(stdev)
        self.plot_perf_graph_v2(y_axis_name, x_axis_name, means_list, stddev_list, filepath_list, label_list,title=title)

    def draw_perf_ratio_diag_from_csvs(self, filepath_list,label_list ,framename, y_axis_name, x_axis_name, title, analyze_mkl=True):
        means_list = []
        stddev_list = []
        dividend = 0
        for filepath in filepath_list:
            means = []
            stdev = []
            ratio_means = []
            ratio_stddev = []
            mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
            if analyze_mkl == True:
                means.append(float(mean1))
                stdev.append(0)
                if dividend == 0:
                    dividend = mean1
                ratio_means.append(float(mean1 / dividend))
            else:
                means.append(float(mean2))
                stdev.append(0)
                if dividend == 0:
                    dividend = mean2
                ratio_means.append(float(mean2 / dividend))
            means_list.append(ratio_means)
            stddev_list.append(stdev)
        self.plot_perf_graph_v2(y_axis_name, x_axis_name, means_list, stddev_list, filepath_list, label_list,title=title)

    def parse_stdout(self, output, keyword):
        output = str(output)
        output = output.split("\n")
        lines = output[0].split('\\')
        for l in lines:
            if l.find("Throughput") > 0:
                return l
        return None

    def read_throughput(self, filepath, keyword='Throughput', index=1):
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    print(line)
                    number = line.split(' ')[index]
                    print(number)
                    return number
        return None

    def read_accuracy(self, filepath, keyword='accuracy', index=-2):
        accuracy=[]
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    number = line.split(' ')[index]
                    number = number.strip(',')
                    number = number.strip('(')
                    number = number.strip(')')
                    accuracy.append(float(number))
        return accuracy

    def read_iteration_time(self, filepath, keyword='Iteration', index=-2):
        iteration=[]
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    number = line.split(' ')[index]
                    number = number.strip(',')
                    number = number.strip('(')
                    number = number.strip(')')
                    iteration.append(float(number))
        return iteration
