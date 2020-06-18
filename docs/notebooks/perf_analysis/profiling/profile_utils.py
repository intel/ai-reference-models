import configparser
import tensorflow as tf
import json
from tensorflow.python.client import timeline
import os
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


class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            self._timeline_dict['traceEvents'] += [event for event in chrome_trace_dict['traceEvents'] if 'ts' in event]

    def update_timeline_from_runmeta(self, run_metadata):
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        self.update_timeline(chrome_trace)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class tfSession(tf.compat.v1.Session):

    def __init__(self, target='', graph=None, config=None, run_metadata=None, many_runs_timeline=None):
        self.run_metadata = run_metadata
        self.options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        self.many_runs_timeline = many_runs_timeline
        super(tfSession, self).__init__(target, graph=graph, config=config)

    def run(self, c, feed_dict=None, options=None, run_metadata=None):
        options = self.options
        run_metadata = self.run_metadata
        ret = super(tfSession, self).run(c, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        self.many_runs_timeline.update_timeline_from_runmeta(run_metadata)
        return ret

    def save_timeline(self, fname):
        self.many_runs_timeline.save(fname)


class TensorflowUtils:

    def is_mkl_enabled(self):
        major_version = int(tf.__version__.split(".")[0])
        if major_version >= 2:
            from tensorflow.python import _pywrap_util_port
            return _pywrap_util_port.IsMklEnabled()
        else:
            return tf.pywrap_tensorflow.IsMklEnabled()


class GitOps:

    def __init__(self, repopath='./'):
        self.repopath = repopath
        if has_git is True:
            self.repo = Repo(repopath)
        return

    def find_commit_with_keyword(self, keyword, search_commits_depth=10):
        ret = False
        if has_git is False:
            return ret
        if not self.repo.bare:
            commits = list(self.repo.iter_commits())[:search_commits_depth]
            for commit in commits:
                if commit.summary.find(keyword) != -1:
                    ret = True
                    return ret
        return ret


class PlatformUtils:

    def __init_(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # let's print CPU information
        print("=" * 20, "CPU Info", "=" * 20)
        # number of cores
        print("Physical cores:", psutil.cpu_count(logical=False))
        print("Total cores:", psutil.cpu_count(logical=True))
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print("Max Frequency:", cpufreq.max)
        print("Min Frequency:", cpufreq.min)
        cpu_socket_count = int(subprocess.check_output(
            'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
        print("Socket Number:", cpu_socket_count)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        self.cpu_socket_count = cpu_socket_count
        self.svmem = svmem


class ConfigFile:

    def __init__(self, confpath='profiling/topo.ini'):
        self.benchmark_only = True
        self.configpath = confpath
        self.wget = ''
        self.in_graph = ''
        self.tf_util = TensorflowUtils()
        self.json_fname = ''
        if self.tf_util.is_mkl_enabled() is True:
            self.json_fname = 'mkl_'
        else:
            self.json_fname = 'stock_'
        self.patches = ''
        self.patched = False
        self.patches_keyword = ''

    def read_section(self):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        return config.sections()

    def read_value_from_section(self, topo_name, key):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        for each_section in config.sections():
            if each_section == topo_name:
                for each_key, each_val in config.items(each_section):
                    if each_key == key:
                        return each_val
        return None

    def read_config(self, topo_name):
        configs = []
        config = configparser.ConfigParser()
        config.read(self.configpath)
        for each_section in config.sections():
            if each_section == topo_name:
                for each_key, each_val in config.items(each_section):
                    key = '--' + each_key
                    if each_key == 'data-location':
                        if each_val is not None:
                            self.benchmark_only = False
                    elif each_key == 'throughput-keyword':
                        if each_val is not None:
                            self.throughput_keyword = each_val
                    elif each_key == 'wget':
                        if each_val is not None:
                            self.wget = each_val
                    elif each_key == 'in-graph':
                        if each_val != '':
                            configs.append(key)
                            configs.append(each_val)
                        self.in_graph = each_val
                    elif each_key == 'json-fname':
                        if each_val is not None:
                            self.json_fname = self.json_fname + each_val
                    elif each_key == 'patches':
                        if each_val is not None:
                            self.patches = each_val
                            val = each_val.split('.')[0]
                            val = val.split('-')[1:]
                            keyword = " ".join(val)
                            self.patches_keyword = keyword
                    else:
                        configs.append(key)
                        configs.append(each_val)

        return configs

    def get_parameters(
            self, topo_name, configvals, batch_size=1, thread_number=1,
            socket_number=1, num_inter_threads=0, num_intra_threads=0):
        benchmark_argvs = []
        benchmark_argvs = benchmark_argvs + configvals
        benchmark_argvs.append('--framework')
        benchmark_argvs.append('tensorflow')
        if batch_size > 0:
            benchmark_argvs.append('--batch-size')
            benchmark_argvs.append(str(batch_size))
        if self.benchmark_only is True:
            benchmark_argvs.append('--benchmark-only')

        if num_inter_threads > 0:
            benchmark_argvs.append('--num_inter_threads=' + str(num_inter_threads))
            benchmark_argvs.append('--data_num_inter_threads=' + str(num_inter_threads))
        else:
            benchmark_argvs.append('--num_inter_threads=' + str(socket_number))
            benchmark_argvs.append('--data_num_inter_threads=' + str(socket_number))

        if num_intra_threads > 0:
            benchmark_argvs.append('--num_intra_threads=' + str(num_intra_threads))
            benchmark_argvs.append('--data_num_intra_threads=' + str(num_intra_threads))
        else:
            benchmark_argvs.append('--num_intra_threads=' + str(thread_number))
            benchmark_argvs.append('--data_num_intra_threads=' + str(thread_number))

        if thread_number > 0:
            benchmark_argvs.append('--num-cores=' + str(thread_number))
        if socket_number == 1:
            benchmark_argvs.append('--socket-id')
            benchmark_argvs.append('0')

        return benchmark_argvs

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

    def patch_model_to_enable_timeline(self, patch_fd="profiling/patches", repopath="../../"):
        if self.patches == '':
            return
        self.git = GitOps(repopath=repopath)
        print("patch keyword: ", self.patches_keyword)
        if self.git.find_commit_with_keyword(self.patches_keyword) is True:
            self.patched = True
            print("has been patched, no action")
            return
        print(self.patches)
        patch_path = patch_fd + os.sep + self.patches
        if os.path.exists(patch_path) is True:
            cmd = "git am " + patch_path
            print(cmd)
            os.system(cmd)
            self.patched = True
        return

    def unpatch_model_to_enable_timeline(self, model_path='../../../models/'):
        if self.patched is False:
            print("has not been patched, no action")
            return
        print("unpatch keyword: ", self.patches_keyword)
        if self.git.find_commit_with_keyword(self.patches_keyword, search_commits_depth=2) is True:
            import os
            print("do unpatch")
            cmd = "git reset HEAD^ "
            print(cmd)
            os.system(cmd)
            cmd = "git checkout " + model_path + "*"
            print(cmd)
            os.system(cmd)
        return


class PerfPresenter:

    def __init__(self, showAbsNumber=False):
        self.tf_util = TensorflowUtils()
        self.showAbsNumber = showAbsNumber
        pass

    def read_throughput(self, topo_name, filepath):
        print(topo_name)
        if topo_name == 'resnet50':
            return self.read_resnet50_throughput(filepath)
        if topo_name == 'wide_deep':
            return self.read_wide_deep_throughput(filepath)

    def read_resnet50_throughput(self, filepath):
        print('in resnet50')
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find('Throughput') != -1:
                    print(line)
                    number = line.split(' ')[1]
                    print(number)
                    return number

    def read_wide_deep_throughput(self, filepath):
        print('in wide deep')
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find('Throughput') != -1:
                    print(line)
                    number = line.split(' ')[3]
                    print(number)
                    return number

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
        ax.set_title('stock TF vs Intel TF')
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
        ax.set_ylabel('Throughput Speeup ')
        ax.set_title('stock TF vs Intel TF')
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
            fnames = ['mkl', 'elapsed_time']
        else:
            fnames = ['mkl', 'elapsed_time', 'throughput']
        if os.path.isfile(filename):
            print('file exists')
        else:
            f = open(filename, 'w')
            with f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()

    def log_infer_perfcsv(self, elapsed_time, throughput, filename):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['mkl', 'elapsed_time', 'throughput']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'mkl': self.tf_util.is_mkl_enabled(),
                 'elapsed_time': elapsed_time,
                 'throughput': throughput})

    def log_train_perfcsv(self, elapsed_time, filename):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['mkl', 'elapsed_time']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'mkl': self.tf_util.is_mkl_enabled(),
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
                if row['mkl'] == 'True':
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
        rects1 = ax.bar(x - width / 2, stock_number, width, yerr=stock_stdev, label='stock TF')
        rects2 = ax.bar(x + width / 2, intel_number, width, yerr=intel_stdev, label='intel TF')
        ax.set_ylabel(ylabel)
        ax.set_title('stock TF vs Intel TF')
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
            print("ERROR. Users must run the benchmark with both Stock TF and Intel TF\n")
            return
        intel_ratio_means.append(float(mean1 / mean2))
        stock_stdev.append(float(stdev2 / mean2))
        intel_stdev.append(float(stdev1 / mean1))
        self.plot_perf_graph(y_axis_name, title, stock_ratio_means, intel_ratio_means, stock_stdev, intel_stdev)

    def parse_stdout(self, output, keyword):
        output = str(output)
        output = output.split("\n")
        lines = output[0].split('\\')
        for l in lines:
            if l.find("Throughput") > 0:
                return l
        return None

    def read_throughput(self, filepath, keyword='Throughput'):
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    print(line)
                    number = line.split(' ')[1]
                    print(number)
                    return number
        return None


class TFTimelinePresenter:

    def __init__(self, showAbsNumber=False):
        self.showAbsNumber = showAbsNumber
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1500)
        self.pd = pd

    def show(self, diag, Type):
        import matplotlib.pyplot as plt
        ret = None
        if self.showAbsNumber is True or Type == 'pie':
            ret = diag
            plt.show()
        plt.clf()
        return ret

    def read_timeline(self, fn, maxents=0):
        import json
        import pandas as pd
        with open(fn, 'r') as (f):
            j = json.loads(f.read())
        allkeys = [list(js.keys()) for js in j['traceEvents']]
        allkeys = [item for sublist in allkeys for item in iter(sublist)]
        allkeys = sorted(set(allkeys))
        argkeys = [js['args'].keys() for js in j['traceEvents'] if 'args' in js]
        argkeys = sorted(set([item for sublist in argkeys for item in iter(sublist)]))
        argkeys = ['arg_' + k for k in argkeys]
        entries = []
        for i, e in enumerate(j['traceEvents']):
            if maxents != 0 and i > maxents:
                break
            ent = {}
            for k, v in e.items():
                if k == 'args':
                    for a in v.keys():
                        ent['arg_' + a] = str(v[a])

                else:
                    ent[k] = str(v)

            entries.append(ent)

        df = pd.DataFrame(entries)
        return df

    def summarize_item(self, tl, item, ascending=False):
        return tl.groupby([item])['dur'].sum().sort_values(ascending=ascending)

    def summarize_barh(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.barh(
            ax=ax, title=title, figsize=figsize, logx=False)
        return ret

    def summarize_pie(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.pie(
            ax=ax, title=title, figsize=figsize, logx=logx, autopct='%1.1f%%')
        return ret

    def opname(self, x):
        return self.demangle(x.split('/')[(-1)])

    def opbase(self, x):
        return self.opname(x).split(':')[(-1)]

    def blockname(self, x):
        return x.split('/')[0].split('_')[0]

    def postprocess_timeline(self, t):
        t['dur'] = t['dur'].astype('float') / 1000
        t['ts'] = t['ts'].astype('float') - t['ts'].astype('float').min()
        t['arg_name'] = t['arg_name'].astype('str').replace('nan', '')
        t['arg_opname'] = t['arg_name'].apply(self.opname)
        t['arg_opbase'] = t['arg_name'].apply(self.opbase)
        return t

    def demangle(self, x, short=True):
        import cxxfilt
        z = x.split('::')
        try:
            if short:
                return z[0] + '::' + cxxfilt.demangle(z[1]).split('<')[0]
            else:
                return z[0] + '::' + cxxfilt.demangle(z[1])
        except IndexError:
            return x

    def get_tf_ops_time(self, timeline_pd, fn, tfile_prefix):
        sitems = self.summarize_item(timeline_pd, 'arg_op')
        import csv
        filename = fn.split('.')[0] + '.csv'
        f = open(filename, 'w')
        time_col_name = 'elapsed_time_' + tfile_prefix
        with f:
            fnames = ['op', time_col_name, 'speedup', 'mkl_op']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writeheader()
            x = 0
            for sitem in sitems:
                mkl_op = False
                if tfile_prefix == 'mkl':
                    op_name = sitems.index[x].strip('_')
                    if op_name.find('Mkl') != -1:
                        mkl_op = True
                        op_name = op_name[3:]
                else:
                    op_name = sitems.index[x].strip('_')
                writer.writerow({'op': op_name, time_col_name: sitems[x], 'speedup': 0, 'mkl_op': mkl_op})
                x = x + 1
        ret = None
        if self.showAbsNumber is True:
            ret = sitems
        return ret

    def plot_summary_barth(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_bar.png'
        title_ = tfile_prefix + 'TF : op duration bar chart'
        ax = self.summarize_barh(timeline_pd, 'arg_op', title=title_, topk=50, logx=True, figsize=(10,
                                                                                                   10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def plot_summary_pie(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_pie.png'
        title_ = tfile_prefix + 'TF : op duration pie chart'
        timeline_pd_known = timeline_pd[(~timeline_pd['arg_op'].str.contains('unknown'))]
        ax = self.summarize_pie(timeline_pd_known, 'arg_op', title=title_, topk=50, logx=True, figsize=(10,
                                                                                                        10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def merge_two_csv_files(self, merged_filepath, a, b):
        merged = a.merge(b, on='op')
        merged['speedup'] = merged['elapsed_time_stock'] / merged['elapsed_time_mkl']
        if merged['mkl_op_x'] is True:
            merged['mkl_op'] = True
        merged['mkl_op'] = merged['mkl_op_x'] + merged['mkl_op_y']
        merged = merged.drop(columns=['speedup_x', 'speedup_y'])
        merged = merged.drop(columns=['mkl_op_x', 'mkl_op_y'])
        if self.showAbsNumber is False:
            ret = merged.drop(columns=['elapsed_time_stock', 'elapsed_time_mkl'])
        else:
            ret = merged
        merged.to_csv(merged_filepath, index=False)
        return ret

    def compare_bar_pie_charts(self, chart_type):
        import matplotlib.pyplot as plt
        import matplotlib.image as img
        from matplotlib import rcParams
        import os
        if chart_type == 'bar':
            imgfiles = [x for x in os.listdir('.') if '_tf_op_duration_bar.png' == x[-23:]]
        elif chart_type == 'pie':
            imgfiles = [x for x in os.listdir('.') if '_tf_op_duration_pie.png' == x[-23:]]
        else:
            return
        rcParams['figure.figsize'] = (30, 30)
        fig, ax = plt.subplots(1, 2)
        index = 0
        for imgf in imgfiles:
            image = img.imread(imgf)
            ax[index].imshow(image)
            index = index + 1

    def plot_compare_bar_charts(self, fpath):
        if self.showAbsNumber is False:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        import csv
        reader = csv.DictReader(open(fpath))
        xlabels = []
        stock_means = []
        intel_means = []
        item_name = reader.fieldnames[0]
        a_name = reader.fieldnames[1]
        b_name = reader.fieldnames[2]

        if a_name.find('mkl') != -1:
            intel_name = a_name
            stock_name = b_name
        else:
            intel_name = b_name
            stock_name = a_name

        for row in reader:
            if row['op'] != 'unknown':
                xlabels.append(row[item_name] + "_(mkl-" + str(row['mkl_op']) + ')')
                intel_means.append(float(row[intel_name]))
                stock_means.append(float(row[stock_name]))

        N = len(xlabels)
        ind = np.arange(N)
        width = 0.35
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, stock_means, width, color='orange')
        rects2 = ax.bar(ind + width, intel_means, width, color='royalblue')
        ax.set_ylabel('Elpased Time (ms)')
        ax.set_title('TF Ops Performance Comparison ')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
        ax.legend((rects1[0], rects2[0]), ['stock', 'intel'], fontsize=20)
        filename = 'compared_tf_op_duration_bar.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_compare_ratio_bar_charts(self, fpath):
        import numpy as np
        import matplotlib.pyplot as plt
        import csv
        reader = csv.DictReader(open(fpath))
        xlabels = []
        stock_means = []
        intel_means = []
        item_name = reader.fieldnames[0]
        c_name = reader.fieldnames[3]
        for row in reader:
            if row['op'] != 'unknown':
                xlabels.append(row[item_name] + "_(mkl-" + str(row['mkl_op']) + ')')
                intel_means.append(float(row[c_name]))
                stock_means.append(1)

        N = len(xlabels)
        ind = np.arange(N)
        width = 0.35
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        rects2 = ax.bar(ind + width / 2, intel_means, width, color='royalblue')
        ax.set_ylabel('Speedup')
        ax.set_title('TF Ops Performance Comparison ')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
        ax.legend([rects2[0]], ['intel'], fontsize=20)
        plt.axhline(y=1, linewidth=4, color='r')
        filename = 'compared_tf_op_duration_ratio_bar.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_compare_pie_charts(self, fpath):
        import matplotlib.pyplot as plt
        import csv
        reader = csv.DictReader(open(fpath))
        xlabels = []
        stock_means = []
        intel_means = []
        item_name = reader.fieldnames[0]
        a_name = reader.fieldnames[1]
        b_name = reader.fieldnames[2]

        if a_name.find('mkl') != -1:
            intel_name = a_name
            stock_name = b_name
        else:
            intel_name = b_name
            stock_name = a_name

        for row in reader:
            if row['op'] != 'unknown':
                xlabels.append(row[item_name])
                intel_means.append(float(row[intel_name]))
                stock_means.append(float(row[stock_name]))

        fig = plt.figure(figsize=(18, 15))

        ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax1.pie(
            stock_means, autopct='%1.1f%%',
            textprops=dict(color="w"), radius=1.2)

        ax1.set_title('Stock Tensorflow ')

        ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax2.pie(
            intel_means, autopct='%1.1f%%',
            textprops=dict(color="w"), radius=1.2)
        ax2.set_title('Intel Tensorflow ')
        ax2.legend(
            wedges,
            xlabels,
            title="TF Ops",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))
        filename = 'compared_tf_op_duration_pie.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()
