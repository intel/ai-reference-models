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
import unittest


class RunCmd:

    def get_run_cmd(self,use_runipy=False):
        if use_runipy is True:
            run_cmd = ' runipy '
            output_cmd = ' '
            extra_cmd = ' '
        else:
            run_cmd = ' jupyter nbconvert --to notebook --execute '
            output_cmd = ' --output '
            extra_cmd = ' --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=1000 '
            #  ACCURACY_ENABLE=False TOPO_INDEX=0   TF_ENABLE_MKL_NATIVE_FORMAT=1  ENABLE_TIMELINE=True jupyter nbconvert --to notebook --execute  benchmark_perf_comparison.ipynb  --output  profiling/unit_test_results/benchmark_perf_comparison_resnet50_infer_fp32.ipynb --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=1000 --ExecutePreprocessor.kernel_name='stock-tensorflow'
        return run_cmd, output_cmd, extra_cmd

    def run(self, cmd):
        import subprocess

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        print("Command exit status/return code : ", p_status)
        return p_status

    def runJupyterNotebook(self, input_notebook_filename, output_notebook_filename, conda_env, fdpath='./'):
        import nbformat
        import os
        from nbconvert.preprocessors import ExecutePreprocessor
        from nbconvert.preprocessors import CellExecutionError
        if os.path.isfile(input_notebook_filename) is False:
            print("No Jupyter notebook found : ",input_notebook_filename)
        try:
            with open(input_notebook_filename) as f:
                nb = nbformat.read(f, as_version=4)
                ep = ExecutePreprocessor(timeout=6000, kernel_name=conda_env, allow_errors=True)
                ep.preprocess(nb, {'metadata': {'path': fdpath}})
                with open(output_notebook_filename, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                return 0
        except CellExecutionError:
            print("Exception!")
            return -1

    def runPerfComparisonCmd(self, conda_env, enable_accuracy, topo_index, enable_timeline, postfix, model_source_dir='', data_download_path=''):
        import os
        os.environ['ACCURACY_ENABLE']=enable_accuracy
        os.environ['TOPO_INDEX']=topo_index
        os.environ['TF_ENABLE_MKL_NATIVE_FORMAT']='1'
        os.environ['ENABLE_TIMELINE']=enable_timeline
        os.environ['MODEL_SOURCE_DIR']=model_source_dir
        os.environ['DATA_DOWNLOAD_PATH']=data_download_path
        cmd = self.runJupyterNotebook('benchmark_perf_comparison.ipynb', 'profiling/unit_test_results/benchmark_perf_comparison_' + postfix + '.ipynb', conda_env)
        return cmd

    def runPerfTimelneCmd(self, conda_env, postfix, fd_inex='0', file_index='0', baseline_index='0'):
        import os
        os.environ['FD_INDEX']=fd_inex
        os.environ['FILE_INDEX']=file_index
        os.environ['BASELINE_INDEX']=baseline_index
        cmd = self.runJupyterNotebook('benchmark_perf_timeline_analysis.ipynb', 'profiling/unit_test_results/benchmark_perf_timeline_analysis_' + postfix + '.ipynb', conda_env)
        return cmd

    def runDataTypeCmd(self, conda_env, enable_accuracy, topo_index, data_type_index, enable_timeline, postfix, model_source_dir='', data_download_path=''):
        import os
        os.environ['ACCURACY_ENABLE']=enable_accuracy
        os.environ['TOPO_INDEX']=topo_index
        os.environ['TF_ENABLE_MKL_NATIVE_FORMAT']='0'
        os.environ['ENABLE_TIMELINE']=enable_timeline
        os.environ['DATA_TYPE_INDEX']=str(data_type_index)
        os.environ['MODEL_SOURCE_DIR']=model_source_dir
        os.environ['DATA_DOWNLOAD_PATH']=data_download_path
        cmd = self.runJupyterNotebook('benchmark_data_types_perf_comparison.ipynb', 'profiling/unit_test_results/benchmark_data_types_perf_comparison_' + postfix + '.ipynb', conda_env)
        return cmd

    def runDataTypeTimelneCmd(self, conda_env, postfix, fd_inex='0', jit_fd_index='0', baseline_index='0', comparison_index='1'):
        import os
        os.environ['FD_INDEX']=fd_inex
        os.environ['JIT_FD_INDEX']=jit_fd_index
        os.environ['BASELINE_INDEX']=baseline_index
        os.environ['COMPARISON_INDEX']=comparison_index
        cmd = self.runJupyterNotebook('benchmark_data_types_perf_timeline_analysis.ipynb', 'profiling/unit_test_results/benchmark_data_types_perf_timeline_analysis_' + postfix + '.ipynb', conda_env)
        return cmd


class Utils:

    def found_files_in_folder(self, pattern, path):
        import os
        import fnmatch
        listOfFiles = os.listdir(path)
        foundfiles = []
        founpaths = []
        for f in listOfFiles:
            if fnmatch.fnmatch(f, pattern):
                foundfiles.append(f)
                founpaths.append(path + os.sep + f)
        return foundfiles, founpaths

    def createFolderForResults(self):
        import os
        path = os.getcwd()
        unit_test_dir = path + os.sep + 'profiling/unit_test_results'
        if os.path.isdir(unit_test_dir) is False:
            os.mkdir(unit_test_dir)
        return unit_test_dir

    def mvTimeline2Results(self):
        import shutil
        import os
        pattern = "Timeline_*"
        path = os.getcwd()
        unit_test_dir = self.createFolderForResults()
        timeline_files, timeline_paths = self.found_files_in_folder(pattern, path)
        for f in timeline_files:
            if os.path.isdir(f) is True:
                print(f)
                shutil.move(f, unit_test_dir)

    def mvCSV2Results(self):
        import shutil
        import os
        pattern = "*.csv"
        path = os.getcwd()
        unit_test_dir = self.createFolderForResults()
        csv_files, csv_paths = self.found_files_in_folder(pattern, path)
        for f in csv_files:
            if os.path.isfile(f) is True:
                print(f)
                perf_ratio = self.get_perf_ratio_from_csv(f, 'throughput')
                model = f.split('.')[0].split('_')[1]
                print(model)
                self.log_model_speedupcsv(model, perf_ratio, 'profiling/unit_test_results/unit_test_summary.csv')
                shutil.move(f, unit_test_dir)

    def get_perf_ratio_from_csv(self, filepath, framename):
        from profiling.profile_utils import PerfPresenter
        perfp = PerfPresenter()
        mean1, mean2, len1, len2, stdev1, stdev2 = perfp.read_number_from_csv(filepath, framename)
        if mean1 == 0 or mean2 == 0:
            print("ERROR. Users must run the benchmark with both Stock TF and Intel TF\n")
            return
        intel_ratio_means = float(mean1 / mean2)
        return intel_ratio_means

    def create_csv_logfile(self, filename):
        import csv
        import os.path
        fnames = ['model', 'speedup']
        if os.path.isfile(filename):
            print('file exists')
        else:
            f = open(filename, 'w')
            with f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()

    def log_model_speedupcsv(self, model, speedup, filename):
        import csv
        import os
        if os.path.isfile(filename) is False:
            self.create_csv_logfile(filename)
        f = open(filename, 'a')
        with f:
            fnames = ['model', 'speedup']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'model': model,
                 'speedup': speedup})

    def test_patches(self, topology_name, ModelZooRoot=""):
        import os
        if ModelZooRoot == "":
            ModelZooRoot= os.getcwd() + "/../../../"
        print("\n start patching ",topology_name)
        from profiling.profile_utils import ConfigFile
        config = ConfigFile()
        config.read_config(topology_name)
        ret1 = config.patch_model_to_enable_timeline(repopath=ModelZooRoot)
        ret2 = config.unpatch_model_to_enable_timeline(model_path=ModelZooRoot+'/models/')
        ret = ret1 or ret2
        print("\n finish patching ",topology_name)
        print("\n Patch status ", ret)
        if ret is False:
            return -1
        else: 
            return 0


class TestPatches(unittest.TestCase):

    def setUp(self):
        import sys
        from profiling.profile_utils import ConfigFile
        import os

        config = ConfigFile()
        self.sections = config.read_supported_section(on_mkl=True)
        print("Supported topologies: ")
        self.index =0 
        for section in self.sections:
            print(" %d: %s " %(self.index, section))
            self.index+=1
        current_path = os.getcwd()
        self.ModelZooRoot = current_path + "/../../../"
        self.marked_out_patches = []

    def tearDown(self):
        print("tear down")

    def test_patche_0(self):
        topo_index = 0
        marked = False
        ret = 0
        for section in self.sections:
            for marked_patch in self.marked_out_patches:
                if section == marked_patch:
                    marked = True
                    print("\n no patch ", section)
            if marked is False:
                cmd_ret = Utils().test_patches(section)
                ret = ret or cmd_ret
            marked = False

        self.assertEqual(ret, 0)


class TestPerfComparison(unittest.TestCase):

    def setUp(self):
        self.TimelineEnabled = True
        self.baseline_conda = 'stock-tensorflow'
        self.compare_conda = 'intel-tensorflow'

        self.rfcn_infer_fp32_model_path = '/tmp/TF-Models/tensorflow-models-rfcn '
        self.rfcn_infer_fp32_data_path = '/tf_dataset/dataset/coco_dataset/raw-data/val2017 '
        self.mobilenet_v1_infer_fp32_model_path = '/tmp/TF-Models/models '
        self.wide_deep_large_infer_fp32_data_path = '/tf_dataset/dataset/wide_deep_kaggle/eval_preprocessed_eval.tfrecords'
        self.bert_infer_fp32_data_path = '/tf_dataset/dataset/bert_official/MRPC '
        self.bert_infer_fp32_model_path = '/tmp/TF-Models/bert '
        self.transformer_mlperf_train_fp32_data_path = '/tf_dataset/dataset/transformer_data '
        self.ssd_mobilenet_infer_fp32_data_path = '/tf_dataset/dataset/SSDMobilenet/coco_val.record '

        Utils().createFolderForResults()

    def tearDown(self):
        self.TimelineEnabled = False
        Utils().mvTimeline2Results()

    def test_0_env_setup(self):
        import os
        env_setup = os.environ.get('ENV_SETUP', '')
        if env_setup != '':
            setup = eval(env_setup)
        else:
            setup = False

        ret = 0
        if setup is True:
            cmd1 = 'conda create -n ' + self.baseline_conda + ' python matplotlib ipykernel psutil pandas gitpython'
            ret1 = RunCmd().run(cmd1)
            cmd2 = 'conda run -n ' + self.baseline_conda + ' pip install tensorflow==2.4.0 cxxfilt'
            ret2 = RunCmd().run(cmd2)
            cmd3 = '~/anaconda3/envs/' + self.baseline_conda + '/bin/python  -m ipykernel install --user --name=' + self.baseline_conda
            ret3 = RunCmd().run(cmd3)
            cmd4 = 'conda create -n intel-tensorflow python matplotlib ipykernel psutil pandas gitpython'
            ret4 = RunCmd().run(cmd4)
            cmd5 = 'conda run -n intel-tensorflow pip install intel-tensorflow==2.4.0 cxxfilt'
            ret5 = RunCmd().run(cmd5)
            cmd6 = '~/anaconda3/envs/intel-tensorflow/bin/python  -m ipykernel install --user --name=intel-tensorflow'
            ret6 = RunCmd().run(cmd6)
            ret = ret1 or ret2 or ret3 or ret4 or ret5 or ret6

        self.assertEqual(ret, 0)

    def test_a_resnet50_infer_fp32(self):
        postfix = 'resnet50_infer_fp32'
        topo_index = '0'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_b_resnet50v1_5_infer_fp32(self):
        postfix = 'resnet50v1_5_infer_fp32'
        topo_index = '1'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

#    def test_c_resnet50v1_5_train_fp32(self):
#        Utils().createFolderForResults()
#        postfix = 'resnet50v1_5_train_fp32'
#        topo_index='2'
#        extra = ' DATA_DOWNLOAD_PATH=/tf_dataset/dataset/TF_Imagenet_FullData/ '
#        cmd1 = 'conda run -n stock-tensorflow ACCURACY_ENABLE=False TOPO_INDEX=' + topo_index + extra +' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret1 = RunCmd().run(cmd1)
#        cmd2 = 'conda run -n intel-tensorflow ACCURACY_ENABLE=False TOPO_INDEX=' + topo_index + extra +' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret2 = RunCmd().run(cmd2)
#        cmd3 = 'conda run -n intel-tensorflow FD_INDEX=0 FILE_INDEX=0 BASELINE_INDEX=1 runipy benchmark_perf_timeline_analysis.ipynb profiling/unit_test_results/benchmark_perf_timeline_analysis_'+postfix+'.ipynb'
#        ret3 = RunCmd().run(cmd3)
#        ret = ret1 or ret2 or ret3
#        Utils().mvTimeline2Results()
#        self.assertEqual(ret, 0)

    def test_d_densenet169_infer_fp32(self):
        postfix = 'densenet169_infer_fp32'
        topo_index = '2'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

#    def test_e_bert_large_infer_fp32(self):
#        Utils().createFolderForResults()
#        postfix = 'bert_large_infer_fp32'
#        topo_index='3'
#        extra = ' DATA_DOWNLOAD_PATH=/tf_dataset/dataset/bert_large_wwm/wwm_uncased_L-24_H-1024_A-16 '
#        cmd1 = 'conda run -n stock-tensorflow ACCURACY_ENABLE=False TOPO_INDEX='+topo_index+extra+' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret1 = RunCmd().run(cmd1)
#        cmd2 = 'conda run -n intel-tensorflow ACCURACY_ENABLE=False TOPO_INDEX='+topo_index+extra+' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret2 = RunCmd().run(cmd2)
#        cmd3 = 'conda run -n intel-tensorflow FD_INDEX=0 FILE_INDEX=0 BASELINE_INDEX=0 runipy benchmark_perf_timeline_analysis.ipynb profiling/unit_test_results/benchmark_perf_timeline_analysis_'+postfix+'.ipynb'
#        ret3 = RunCmd().run(cmd3)
#        ret = ret1 or ret2 or ret3
#        Utils().mvTimeline2Results()
#        self.assertEqual(ret, 0)

#    def test_f_bert_large_train_fp32(self):
#        Utils().createFolderForResults()
#        postfix = 'bert_large_train_fp32'
#        topo_index='5'
#        extra = ' DATA_DOWNLOAD_PATH=/tf_dataset/dataset/bert_large_wwm/wwm_uncased_L-24_H-1024_A-16 '
#        cmd1 = 'conda run -n stock-tensorflow ACCURACY_ENABLE=False TOPO_INDEX=' + topo_index + extra +' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret1 = RunCmd().run(cmd1)
#        cmd2 = 'conda run -n intel-tensorflow ACCURACY_ENABLE=False TOPO_INDEX=' + topo_index + extra +' ENABLE_TIMELINE=True runipy benchmark_perf_comparison.ipynb profiling/unit_test_results/benchmark_perf_comparison_'+postfix+'.ipynb'
#        ret2 = RunCmd().run(cmd2)
#        cmd3 = 'conda run -n intel-tensorflow FD_INDEX=0 FILE_INDEX=0 BASELINE_INDEX=1 runipy benchmark_perf_timeline_analysis.ipynb profiling/unit_test_results/benchmark_perf_timeline_analysis_'+postfix+'.ipynb'
#        ret3 = RunCmd().run(cmd3)
#        ret = ret1 or ret2 or ret3
#        Utils().mvTimeline2Results()
#        self.assertEqual(ret, 0)

    def test_g_rfcn_infer_fp32(self):
        postfix = 'rfcn_infer_fp32'
        topo_index = '3'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.rfcn_infer_fp32_model_path, data_download_path=self.rfcn_infer_fp32_data_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.rfcn_infer_fp32_model_path, data_download_path=self.rfcn_infer_fp32_data_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_h_inceptionv3_infer_fp32(self):
        postfix = 'inceptionv3_infer_fp32'
        topo_index = '4'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_i_inceptionv4_infer_fp32(self):
        postfix = 'inceptionv4_infer_fp32'
        topo_index = '5'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_j_mobilenet_v1_infer_fp32(self):
        postfix = 'mobilenet_v1_infer_fp32'
        topo_index = '6'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.mobilenet_v1_infer_fp32_model_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.mobilenet_v1_infer_fp32_model_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_k_resnet101_infer_fp32(self):
        postfix = 'resnet101_infer_fp32'
        topo_index = '7'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_l_wide_deep_large_infer_fp32(self):
        postfix = 'wide_deep_large_infer_fp32'
        topo_index = '8'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, data_download_path=self.wide_deep_large_infer_fp32_data_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, data_download_path=self.wide_deep_large_infer_fp32_data_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_m_bert_infer_fp32(self):
        postfix = 'bert_infer_fp32'
        topo_index = '9'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.bert_infer_fp32_model_path, data_download_path=self.bert_infer_fp32_data_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, model_source_dir=self.bert_infer_fp32_model_path, data_download_path=self.bert_infer_fp32_data_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_n_transformer_mlperf_train_fp32(self):
        postfix = 'transformer_mlperf_train_fp32'
        topo_index = '10'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, data_download_path=self.transformer_mlperf_train_fp32_data_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, data_download_path=self.transformer_mlperf_train_fp32_data_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_o_ssd_mobilenet_infer_fp32(self):
        postfix = 'ssd_mobilenet_infer_fp32'
        topo_index = '11'
        ret1 = RunCmd().runPerfComparisonCmd(self.baseline_conda, 'False', topo_index, 'True', postfix, data_download_path=self.ssd_mobilenet_infer_fp32_data_path)
        ret2 = RunCmd().runPerfComparisonCmd(self.compare_conda, 'False', topo_index, 'True', postfix, data_download_path=self.ssd_mobilenet_infer_fp32_data_path)
        ret3 = RunCmd().runPerfTimelneCmd(self.compare_conda, postfix)
        ret = ret1 or ret2 or ret3
        self.assertEqual(ret, 0)

    def test_z_summarize(self):
        cmd1 = 'conda run -n intel-tensorflow python summarize_unit_test.py -v'
        ret1 = RunCmd().run(cmd1)
        cmd2 = 'ipython nbconvert profiling/unit_test_results/*.ipynb'
        ret2 = RunCmd().run(cmd2)
        ret = ret1 or ret2
        self.assertEqual(ret, 0)


class TestPerfDataType(unittest.TestCase):

    def setUp(self):
        self.TimelineEnabled = 'True'
        self.baseline_conda = 'stock-tensorflow'
        self.compare_conda = 'intel-tensorflow'
        self.mobilenet_v1_infer_model_path = '/tmp/TF-Models/models'
        self.rfcn_infer_model_path = '/tmp/TF-Models/tensorflow-models-rfcn'
        self.rfcn_infer_data_path = '/tf_dataset/dataset/coco_dataset/raw-data/val2017 '
        self.ssdmobilenet_data_path = '/tf_dataset/dataset/SSDMobilenet/coco_val.record'
        self.wide_deep_large_ds_infer_data_path = '/tf_dataset/dataset/wide_deep_kaggle/eval_preprocessed_eval.tfrecords'
        self.transformer_mlperf_training_data_path = '/tf_dataset/dataset/transformer_data'
        Utils().createFolderForResults()

    def tearDown(self):
        self.TimelineEnabled = 'False'
        Utils().mvTimeline2Results()

    def test_0_env_setup(self):
        import os
        env_setup = os.environ.get('ENV_SETUP', '')
        if env_setup != '':
            setup = eval(env_setup)
        else:
            setup = False

        ret = 0
        if setup is True:
            cmd1 = 'conda create -n stock-tensorflow python matplotlib ipykernel psutil pandas gitpython'
            ret1 = RunCmd().run(cmd1)
            cmd2 = 'conda run -n stock-tensorflow pip install tensorflow==2.4.0 cxxfilt'
            ret2 = RunCmd().run(cmd2)
            cmd3 = '~/anaconda3/envs/stock-tensorflow/bin/python  -m ipykernel install --user --name=stock-tensorflow'
            ret3 = RunCmd().run(cmd3)
            cmd4 = 'conda create -n intel-tensorflow python matplotlib ipykernel psutil pandas gitpython'
            ret4 = RunCmd().run(cmd4)
            cmd5 = 'conda run -n intel-tensorflow pip install intel-tensorflow==2.4.0 cxxfilt'
            ret5 = RunCmd().run(cmd5)
            cmd6 = '~/anaconda3/envs/intel-tensorflow/bin/python  -m ipykernel install --user --name=intel-tensorflow'
            ret6 = RunCmd().run(cmd6)
            ret = ret1 or ret2 or ret3 or ret4 or ret5 or ret6

        self.assertEqual(ret, 0)

    def test_a_resnet50v1_5_infer(self):
        postfix = 'resnet50v1_5_infer'
        topo_index = '3'
        ret = 0
        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret

        self.assertEqual(ret, 0)

    def test_b_resnet50_infer(self):
        postfix = 'resnet50_infer'
        topo_index = '2'
        ret = 0
        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret

        self.assertEqual(ret, 0)

    def test_c_inceptionv4_infer(self):
        postfix = 'inceptionv4_infer'
        topo_index = '0'
        ret = 0
        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret
        self.assertEqual(ret, 0)

#    def test_d_mobilenet_v1_infer(self):
#        postfix = 'mobilenet_v1_infer'
#        topo_index = '1'
#        ret = 0

#        data_type_index = 0
#        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, model_source_dir=self.mobilenet_v1_infer_model_path)
#        ret = ret or cmd_ret

#        data_type_index += 1
#        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, model_source_dir=self.mobilenet_v1_infer_model_path)
#        ret = ret or cmd_ret

#        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
#        ret = ret or cmd_ret
#        self.assertEqual(ret, 0)

    def test_e_resnet101_infer(self):
        postfix = 'resnet101_infer'
        topo_index = '1'
        ret = 0

        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret
        self.assertEqual(ret, 0)

    def test_f_rfcn_infer(self):
        postfix = 'rfcn_infer'
        topo_index = '4'
        ret = 0

        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, model_source_dir=self.rfcn_infer_model_path, data_download_path=self.rfcn_infer_data_path)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, model_source_dir=self.rfcn_infer_model_path, data_download_path=self.rfcn_infer_data_path)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret
        self.assertEqual(ret, 0)

    def test_g_ssdmobilenet_infer(self):
        postfix = 'ssdmobilenet'
        topo_index = '5'
        ret = 0
        extra = ' DATA_DOWNLOAD_PATH=' + self.ssdmobilenet_data_path + ' '

        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.ssdmobilenet_data_path)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.ssdmobilenet_data_path)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret
        self.assertEqual(ret, 0)

#    def test_h_wide_deep_large_ds_infer(self):
#        postfix = 'wide_deep_large_ds'
#        topo_index = '8'
#        ret = 0

#        data_type_index = 0
#        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.wide_deep_large_ds_infer_data_path)
#        ret = ret or cmd_ret

#        data_type_index += 1
#        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.wide_deep_large_ds_infer_data_path)
#        ret = ret or cmd_ret

#        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
#        ret = ret or cmd_ret
#        self.assertEqual(ret, 0)

    def test_i_transformer_mlperf_training(self):
        postfix = 'transformer_mlperf'
        topo_index = '6'
        ret = 0

        data_type_index = 0
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.transformer_mlperf_training_data_path)
        ret = ret or cmd_ret

        data_type_index += 1
        cmd_ret = RunCmd().runDataTypeCmd(self.compare_conda, 'False', topo_index, data_type_index, self.TimelineEnabled, postfix, data_download_path=self.transformer_mlperf_training_data_path)
        ret = ret or cmd_ret

        cmd_ret = RunCmd().runDataTypeTimelneCmd(self.compare_conda, postfix)
        ret = ret or cmd_ret
        self.assertEqual(ret, 0)

    def test_z_summarize(self):
        cmd = 'ipython nbconvert profiling/unit_test_results/*.ipynb'
        ret = RunCmd().run(cmd)
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()


#
# validated models
#
# 0: resnet50 infer fp32
# 1: resnet50v1_5 infer fp32
# 2: densenet169 infer fp32
# 3: rfcn infer fp32
# 4: inceptionv3 infer fp32
# 5: inceptionv4 infer fp32
# 6: mobilenet_v1 infer fp32
# 7: resnet101 infer fp32
# 8: wide_deep_large infer fp32
# 9: bert infer fp32
# 10: transformer_mlperf train fp32
# 11: ssd-mobilenet infer fp32


# data type models
# Index 0 : ('inceptionv4', 'inference')
# Index 1 : ('resnet101', 'inference')
# Index 2 : ('resnet50', 'inference')
# Index 3 : ('resnet50v1_5', 'inference')
# Index 4 : ('rfcn', 'inference')
# Index 5 : ('ssd-mobilenet', 'inference')
# Index 6 : ('transformer_mlperf', 'training')
