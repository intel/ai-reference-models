#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
from contextlib import contextmanager
import os
import pytest
import sys
import tempfile

try:
    # python 2
    from cStringIO import StringIO
except ImportError:
    # python 3
    # only supports unicode so can't be used in python 2 for sys.stdout
    # because (from `print` documentation)
    # "All non-keyword arguments are converted to strings like str() does"
    from io import StringIO


from mock import MagicMock, patch

from benchmarks.common.base_model_init import BaseModelInitializer
from benchmarks.common.base_model_init import set_env_var


@contextmanager
def catch_stdout():
    _stdout = sys.stdout
    sys.stdout = caught_output = StringIO()
    try:
        yield caught_output
    finally:
        sys.stdout = _stdout
        caught_output.close()


@pytest.fixture
def mock_json(patch):
    return patch('json')


@pytest.fixture
def mock_glob(patch):
    return patch('glob.glob')


# Example args and output strings for testing mocks
test_model_name = "resnet50"
test_framework = "tensorflow"
test_mode = "inference"
test_precision = "fp32"
test_docker_image = "foo"
example_req_args = ["--model-name", test_model_name,
                    "--framework", test_framework,
                    "--mode", test_mode,
                    "--precision", test_precision,
                    "--docker-image", test_docker_image]


@patch("benchmarks.common.base_model_init.open")
@patch("common.platform_util.os")
@patch("common.platform_util.system_platform")
@patch("common.platform_util.subprocess")
@patch("os.system")
def test_base_model_initializer(
        mock_system, mock_subprocess, mock_platform, mock_os, mock_open):
    # Setup base model init with test settings
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name, batch_size=100,
                     numa_cores_per_instance=None)
    os.environ["PYTHON_EXE"] = "python"
    os.environ["MPI_HOSTNAMES"] = "None"
    os.environ["MPI_NUM_PROCESSES"] = "None"
    base_model_init = BaseModelInitializer(args, [], platform_util)

    # call run_command and then check the output
    test_run_command = "python foo.py"
    base_model_init.run_command(test_run_command)
    mock_system.assert_called_with(test_run_command)


def test_env_var_already_set():
    """ Tests setting and env var when it's already set """
    env_var = "model-zoo-test-env-var-name"
    original_value = "original"
    modified_value = "modified"

    try:
        # Set the env var to an initial value
        os.environ[env_var] = original_value

        # Try to modify that value but set overwrite flag to False
        set_env_var(env_var, modified_value, overwrite_existing=False)

        # Verify that we still have the original value
        assert os.environ[env_var] == original_value

        # Try to modify the value with the overwrite flag set to True
        set_env_var(env_var, modified_value, overwrite_existing=True)

        # Verify that we now have the modified value
        assert os.environ[env_var] == modified_value
    finally:
        if os.environ.get(env_var):
            del os.environ[env_var]


def test_env_var_not_already_set():
    """ Tests setting and env var when it's not already set """
    env_var = "model-zoo-test-env-var-name"
    new_value = "new_value"

    try:
        # Make sure that the env var is unset to start
        if os.environ.get(env_var):
            del os.environ[env_var]

        # Try setting the value with the overwrite flag set to False
        set_env_var(env_var, new_value, overwrite_existing=False)

        # Verify that we now have a value
        assert os.environ[env_var] == new_value

        # Unset the env var and set it with the overwrite flag set to True
        del os.environ[env_var]
        new_value = "another_new_value"
        set_env_var(env_var, new_value, overwrite_existing=True)

        # Verify that we have the new value set
        assert os.environ[env_var] == new_value
    finally:
        if os.environ.get(env_var):
            del os.environ[env_var]


def test_set_kmp_vars_config_json_does_not_exists():
    """Test config.json does not exist"""
    # Setup base model init with test settings
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    os.environ["PYTHON_EXE"] = "python"
    base_model_init = BaseModelInitializer(args, [], platform_util)

    config_file_path = '/test/foo/config.json'

    with catch_stdout() as caught_output:
        base_model_init.set_kmp_vars(config_file_path)
        output = caught_output.getvalue()

    assert "Warning: File {} does not exist and \
            cannot be used to set KMP environment variables".format(config_file_path) == output.strip()


def test_set_kmp_vars_config_json_exists(mock_json):
    """Test config.json when exists"""
    # Setup base model init with test settings
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    os.environ["PYTHON_EXE"] = "python"
    base_model_init = BaseModelInitializer(args, [], platform_util)

    file_descriptor, config_file_path = tempfile.mkstemp(suffix=".json")

    base_model_init.set_kmp_vars(config_file_path)


@pytest.mark.parametrize('precision', ['int8'])
def test_command_prefix_tcmalloc_int8(precision, mock_glob):
    """ For Int8 models, TCMalloc should be enabled by default and models should include
     LD_PRELOAD in the command prefix, unless disable_tcmalloc=True is set """
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    test_tcmalloc_lib = "/usr/lib/libtcmalloc.so.4.2.6"
    mock_glob.return_value = [test_tcmalloc_lib]
    os.environ["PYTHON_EXE"] = "python"
    args.socket_id = 0
    args.num_cores = -1
    args.precision = precision
    args.numa_cores_per_instance = None

    # If tcmalloc is not disabled, we should have LD_PRELOAD in the prefix
    args.disable_tcmalloc = False
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id)
    print(command_prefix)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) in command_prefix
    assert "numactl --cpunodebind=0 --membind=0" in command_prefix

    # If tcmalloc is disabled, LD_PRELOAD shouild not be in the prefix
    args.disable_tcmalloc = True
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) not in command_prefix
    assert "numactl --cpunodebind=0 --membind=0" in command_prefix

    # If numactl is set to false, we should not have numactl in the prefix
    args.disable_tcmalloc = False
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id, numactl=False)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) in command_prefix
    assert "numactl" not in command_prefix


@pytest.mark.parametrize('precision', ['fp32'])
def test_command_prefix_tcmalloc_fp32(precision, mock_glob):
    """ FP32 models should have TC Malloc disabled by default, but models should
    include LD_PRELOAD in the command prefix if disable_tcmalloc=False is explicitly set. """
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    test_tcmalloc_lib = "/usr/lib/libtcmalloc.so.4.2.6"
    mock_glob.return_value = [test_tcmalloc_lib]
    os.environ["PYTHON_EXE"] = "python"
    args.socket_id = 0
    args.num_cores = -1
    args.precision = precision
    args.numa_cores_per_instance = None

    # By default, TCMalloc should not be used
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) not in command_prefix
    assert "numactl --cpunodebind=0 --membind=0" in command_prefix

    # If tcmalloc is disabled, LD_PRELOAD shouild not be in the prefix
    args.disable_tcmalloc = False
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) in command_prefix
    assert "numactl --cpunodebind=0 --membind=0" in command_prefix

    # If numactl is set to false, we should not have numactl in the prefix
    args.disable_tcmalloc = True
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id, numactl=False)
    assert "LD_PRELOAD={}".format(test_tcmalloc_lib) not in command_prefix
    assert "numactl" not in command_prefix


@pytest.mark.parametrize('precision', ['fp32', 'int8', 'bfloat16'])
def test_command_prefix_numa_multi_instance(precision, mock_glob):
    """ Tests that models don't get a numactl command prefix from
    base_model_init.get_command_prefix, if numa_cores_per_instance is set, since numactl
    commands will be added later for each instance run. """
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    test_tcmalloc_lib = "/usr/lib/libtcmalloc.so.4.2.6"
    mock_glob.return_value = [test_tcmalloc_lib]
    os.environ["PYTHON_EXE"] = "python"
    args.socket_id = 0
    args.precision = precision
    args.numa_cores_per_instance = 4

    base_model_init = BaseModelInitializer(args, [], platform_util)
    command_prefix = base_model_init.get_command_prefix(args.socket_id)
    assert "numactl" not in command_prefix


@pytest.mark.skip("Method get_multi_instance_train_prefix() no longer exists")
def test_multi_instance_train_prefix():
    platform_util = MagicMock()
    args = MagicMock(verbose=True, model_name=test_model_name)
    args.num_processes = 2
    args.num_processes_per_node = 1
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command = base_model_init.get_multi_instance_train_prefix(option_list=["--genv:test"])
    assert command == "mpirun -n 2 -ppn 1 --genv test "

    args.num_processes = None
    args.num_processes_per_node = None
    base_model_init = BaseModelInitializer(args, [], platform_util)
    command = base_model_init.get_multi_instance_train_prefix(option_list=["--genv:test", "--genv:test2"])
    assert command == "mpirun --genv test --genv test2 "


@pytest.mark.parametrize('test_num_instances,test_socket_id,test_cpu_list, expected_cpu_bind,precision',
                         [['2', -1, [['0', '1'], ['2', '3']], ['0,1', '2,3'], 'fp32'],
                          ['2', 0, [['0', '1'], ['2', '3']], ['0,1'], 'fp32'],
                          ['2', 1, [['0', '1'], ['2', '3']], ['2,3'], 'fp32'],
                          ['4', -1, [['4', '5', '6', '7', '0', '1', '2', '3']], ['4,5,6,7', '0,1,2,3'], 'fp32'],
                          ['8', -1, [['4', '5', '6', '7', '0', '1', '2', '3']], ['4,5,6,7,0,1,2,3'], 'fp32'],
                          ['3', -1, [['4', '5', '6', '7'], ['0', '1', '2', '3']], ['4,5,6', '7,0,1'], 'fp32'],
                          ['2', -1, [['0', '1'], ['2', '3']], ['0,1', '2,3'], 'int8'],
                          ['2', 0, [['0', '1'], ['2', '3']], ['0,1'], 'int8'],
                          ['2', 1, [['0', '1'], ['2', '3']], ['2,3'], 'int8'],
                          ['4', -1, [['4', '5', '6', '7', '0', '1', '2', '3']], ['4,5,6,7', '0,1,2,3'], 'int8'],
                          ['8', -1, [['4', '5', '6', '7', '0', '1', '2', '3']], ['4,5,6,7,0,1,2,3'], 'int8'],
                          ['3', -1, [['4', '5', '6', '7'], ['0', '1', '2', '3']], ['4,5,6', '7,0,1'], 'int8']])
@patch("os.path.exists")
@patch("glob.glob")
@patch("benchmarks.common.base_model_init.open")
@patch("common.platform_util.os")
@patch("common.platform_util.system_platform")
@patch("common.platform_util.subprocess")
@patch("os.system")
def test_numa_multi_instance_run_command(
        mock_system, mock_subprocess, mock_platform, mock_os, mock_open, mock_glob,
        mock_path_exists, test_num_instances, test_socket_id, test_cpu_list, expected_cpu_bind, precision):
    """ Test the multi instance run using numactl by trying different combinations of
    cpu lists and the number of cores used per instance. Checks the system call that
    is run to verify that it matches the cpu groups that are expected. """
    platform_util = MagicMock(cpu_core_list=test_cpu_list)
    test_output_dir = "/tmp/output"
    args = MagicMock(verbose=True, model_name=test_model_name, batch_size=100,
                     numa_cores_per_instance=test_num_instances, precision=precision,
                     output_dir=test_output_dir, mode="inference", socket_id=test_socket_id,
                     benchmark_only=True)
    os.environ["PYTHON_EXE"] = "python"
    os.environ["MPI_HOSTNAMES"] = "None"
    os.environ["MPI_NUM_PROCESSES"] = "None"
    ld_preload_path = "/usr/lib/libtcmalloc.so.4.2.6"
    mock_glob.return_value = [ld_preload_path]
    base_model_init = BaseModelInitializer(args, [], platform_util)

    mock_path_exists.return_value = True

    expected_omp_num_threads = "OMP_NUM_THREADS={}".format(test_num_instances)

    # tcmalloc is used by default for int8, so the LD_PRELOAD path should be set
    expected_ld_preload = ""
    if precision == "int8":
        expected_ld_preload = "LD_PRELOAD={} ".format(ld_preload_path)

    test_run_command = "python foo.py"

    # For int8, add on the LD_PRELOAD prefix. The run command should know how to move this prefix to the
    # front of the command (before numactl)
    if precision == "int8":
        test_run_command_with_prefix = expected_ld_preload + test_run_command
    else:
        test_run_command_with_prefix = test_run_command

    # call run_command and then check the output
    base_model_init.run_command(test_run_command_with_prefix)
    system_call_args = mock_system.call_args[0][0]

    for cpu_bind in expected_cpu_bind:
        expected_cmd = "{}{} numactl --localalloc --physcpubind={} {} >> {}".\
            format(expected_ld_preload, expected_omp_num_threads, cpu_bind, test_run_command, test_output_dir)
        assert expected_cmd in system_call_args


@pytest.mark.parametrize('test_num_instances,test_socket_id,test_num_cores,test_cpu_list,test_cpuset,'
                         'expected_inter_threads,expected_intra_threads',
                         [[2, -1, -1, [['0', '1'], ['2', '3']], {0: ['0', '1'], 1: ['2', '3']}, 1, 2],
                          [None, 0, -1, [['1', '2', '3'], ['10', '11']], {0: ['1', '2', '3'], 1: ['10', '11']}, 1, 3],
                          [None, 1, -1, [['1', '2', '3'], ['10', '11']], {0: ['1', '2', '3'], 1: ['10', '11']}, 1, 2],
                          [None, 1, -1, [['1', '2', '3'], ['10', '11']], None, 1, 3],
                          [None, 1, 8, [['1', '2', '3'], ['10', '11']], {0: ['1', '2', '3'], 1: ['10', '11']}, 1, 8]])
@patch("os.path.exists")
@patch("benchmarks.common.base_model_init.open")
@patch("common.platform_util.os")
@patch("common.platform_util.system_platform")
@patch("common.platform_util.subprocess")
@patch("os.system")
def test_num_inter_intra_threads_settings(
        mock_system, mock_subprocess, mock_platform, mock_os, mock_open,
        mock_path_exists, test_num_instances, test_socket_id, test_num_cores,
        test_cpu_list, test_cpuset, expected_inter_threads, expected_intra_threads):
    """
    Tests the base model init function that determines the num_inter_threads and
    num_intra_thread values.
    """
    platform_util = MagicMock(cpu_core_list=test_cpu_list, cpuset_cpus=test_cpuset,
                              num_cores_per_socket=len(test_cpu_list[0]))
    test_output_dir = "/tmp/output"
    args = MagicMock(verbose=True, model_name=test_model_name, batch_size=100,
                     numa_cores_per_instance=test_num_instances, precision="fp32",
                     output_dir=test_output_dir, socket_id=test_socket_id, num_cores=test_num_cores,
                     num_inter_threads=None, num_intra_threads=None)
    os.environ["PYTHON_EXE"] = "python"
    os.environ["MPI_HOSTNAMES"] = "None"
    os.environ["MPI_NUM_PROCESSES"] = "None"
    base_model_init = BaseModelInitializer(args, [], platform_util)

    mock_path_exists.return_value = True

    # Get the number of inter/intra threads and compared to the expected values
    base_model_init.set_num_inter_intra_threads()
    print(base_model_init.args.num_inter_threads)
    print(base_model_init.args.num_intra_threads)
    assert base_model_init.args.num_inter_threads == expected_inter_threads
    assert base_model_init.args.num_intra_threads == expected_intra_threads
