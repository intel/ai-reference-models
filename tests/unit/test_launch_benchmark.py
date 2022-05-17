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
from __future__ import print_function
from conditional import conditional

import os
import sys

import pytest
from mock import MagicMock, patch as mock_patch
from test_utils import platform_config
from launch_benchmark import LaunchBenchmark


# Example args and output strings for testing mocks
test_model_name = "resnet50"
test_framework = "tensorflow"
test_mode = "inference"
test_precision = "fp32"
test_docker_image = "foo"
test_batch_size = "100"
test_num_cores = "1"
# need a valid file for tests to work, see conftest.py for where this is managed
test_input_graph = "test.pb"
test_tfserving_framework = "tensorflow_serving"


def setup_mock_values(mock_platform):
    platform_config.set_mock_system_type(mock_platform)


@pytest.fixture
def mock_platform_util(patch):
    return patch("common.base_benchmark_util.platform_util.PlatformUtil",
                 MagicMock(num_cpu_sockets=1, num_cores_per_socket=1, num_threads_per_core=1, num_local_cpus=1,
                           num_numa_nodes=1))


@pytest.fixture
def mock_os(patch):
    return patch("common.base_benchmark_util.platform_util.os")


@pytest.fixture
def mock_subprocess(patch):
    return patch("common.base_benchmark_util.platform_util.subprocess")


@pytest.fixture
def mock_popen(patch):
    return patch("subprocess.Popen")


@pytest.fixture
def mock_system_platform(patch):
    return patch("common.base_benchmark_util.platform_util.system_platform")


@pytest.fixture
def platform_mock(patch):
    return patch("system_platform.system")


@pytest.fixture
def mock_path_exists(patch):
    return patch("os.path.exists", MagicMock(return_value=True))


@pytest.fixture
def mock_isfile(patch):
    return patch("os.path.isfile", MagicMock(return_value=True))


@pytest.fixture
def mock_isdir(patch):
    return patch("os.path.isdir", MagicMock(return_value=True))


@pytest.fixture
def mock_islink(patch):
    return patch("os.path.islink", MagicMock(return_value=False))


@pytest.fixture
def mock_stat(patch):
    stat = MagicMock()
    stat.return_value.st_nlink = 0
    return patch("os.stat", stat)


@pytest.fixture(autouse=True)
def launch_benchmark(mock_platform_util, request, mock_isdir, mock_isfile, mock_islink, mock_stat, mock_path_exists):
    """sets up launch_benchmark obj for every test case and handles catching errors if we wanna test that
       To catch errors called when running launch_benchmark, use something like:
           ['catch_error', SystemExit, [{args}], {error_message}] in parametrize
       where args are args to pass to the benchmark creation and error_message is an optional error message to check for
       otherwise just pass in the req args you'd like to run with via []
       catch_error_override_all_params will not use any example_req_args when creating benchmark

       Sample request.params:
       ['catch_error', SystemExit, []]
       ['catch_error_override_all_params', SystemExit, []]
       ['catch_error', SystemExit, ['--framework', 'foo'], "The specified framework is not supported"]]
       """
    catch_error = False
    error = None
    error_message = ''

    # deleting from this sometimes so need to redeclare it, probably can do that differently...
    example_req_args = ["--model-name", test_model_name,
                        "--framework", test_framework,
                        "--mode", test_mode,
                        "--precision", test_precision,
                        "--docker-image", test_docker_image,
                        "--batch-size", test_batch_size,
                        "--num-cores", test_num_cores]

    if hasattr(request, 'param'):
        if 'catch_error' in request.param[0]:
            catch_error = True
            error = request.param[1]
            if request.param[0] != 'catch_error_override_all_params':
                # TODO: make more efficient! Want to get rid of any example_req_args that exist in request.param[2]
                # using safe deletion from the back
                for idx in range(len(example_req_args) - 1, -1, -1):
                    arg = example_req_args[idx]
                    if not arg.startswith('--'):
                        continue
                    if arg in request.param[2]:
                        # flags are always followed by their value in example_req_args, so delete both arg and its value
                        del example_req_args[idx]
                        del example_req_args[idx]
                req_args = request.param[2] + example_req_args
            else:
                req_args = request.param[2]
            error_message = request.param[3] if len(request.param) == 4 else ''
        else:
            # add extra arguments to the default ones when calling LaunchBenchmark
            req_args = request.param + example_req_args
    else:
        # only use default arguments when calling LaunchBenchmark
        req_args = example_req_args

    with mock_patch.object(sys, "argv", ['run_tf_benchmark.py'] + req_args):
        with conditional(catch_error, pytest.raises(error)) as e:
            obj = LaunchBenchmark(mock_platform_util)
            if error_message:
                assert error_message in str(e.value)
            return obj


def test_launch_benchmark_parse_args(launch_benchmark):
    """
    Verifies that that arg parsing gives us the expected results.
    """
    assert launch_benchmark.args.model_name == test_model_name
    assert launch_benchmark.args.framework == test_framework
    assert launch_benchmark.args.mode == test_mode
    assert launch_benchmark.args.precision == test_precision
    assert launch_benchmark.args.docker_image == test_docker_image
    assert launch_benchmark.unknown_args == []


@pytest.mark.parametrize('launch_benchmark', [["--test", "foo"]], indirect=True)
def test_launch_benchmark_parse_unknown_args(launch_benchmark):
    """
    Checks parsing of unknown args
    """
    assert launch_benchmark.unknown_args == ["--test"]


@pytest.mark.parametrize('launch_benchmark', [['catch_error_override_all_params', SystemExit, []],
                                              ['catch_error', SystemExit, ['--framework', 'foo'],
                                                  "The specified framework is not supported"],
                                              ['catch_error', SystemExit, ['--docker-image', 'test '],
                                                  "docker image string should not have whitespace(s)"],
                                              ['catch_error', ValueError, ["--model-name", test_model_name,
                                                                           "--framework", test_framework,
                                                                           "--mode", "training",
                                                                           "--precision", test_precision,
                                                                           "--docker-image", test_docker_image,
                                                                           "--benchmark-only",
                                                                           "--output-results"],
                                                  "--output-results can only be used when running "
                                                  "inference with a dataset"],
                                              ['catch_error', ValueError, ["--model-name", test_model_name,
                                                                           "--framework", test_framework,
                                                                           "--mode", "training",
                                                                           "--precision", test_precision,
                                                                           "--docker-image", test_docker_image,
                                                                           "--accuracy-only",
                                                                           "--output-results"],
                                                  "--output-results can only be used when running "
                                                  "inference with a dataset"],
                                              ['catch_error_override_all_params', SystemExit,
                                               ["--model-name", test_model_name,
                                                "--framework", test_framework,
                                                "--mode", test_mode,
                                                "--precision", test_precision,
                                                "--volume", "~:test"],
                                                  "Volume mounts can only be used when running in a docker container"]
                                              ], indirect=True)
def test_launch_benchmark_parse_bad_args(launch_benchmark):
    """
    Checks for failures with no args or bad args
    """
    pass


@pytest.mark.parametrize('launch_benchmark', [["--model-name", test_model_name,
                                               "--framework", test_framework,
                                               "--mode", "training",
                                               "--precision", test_precision,
                                               "--docker-image", test_docker_image,
                                               "--data-location", ".",
                                               "--benchmark-only",
                                               "--output-results"]])
def test_output_results_with_accuracy(launch_benchmark, mock_system_platform, mock_os, mock_subprocess):
    """
    Tests that the launch script validation passes when running accuracy with output
    """
    pass


def test_launch_benchmark_validate_model(launch_benchmark, mock_popen, platform_mock):
    """
    Verifies that a valid model name passes validation and starts a docker container.
    """
    platform_mock.return_value = platform_config.OS_TYPE
    launch_benchmark.main()
    assert mock_popen.called
    args, kwargs = mock_popen.call_args
    assert "docker" == args[0][0]
    assert "run" == args[0][1]

@pytest.mark.parametrize("os_type", [["Linux"],
                                     ["Windows"]])
def test_bare_metal(launch_benchmark, mock_popen, platform_mock, os_type):
    """ Tests the bare metal launch script function """
    platform_mock.return_value = platform_config.OS_TYPE
    test_env_vars = {"TEST_ENV_VAR_1": "a", "TEST_ENV_VAR_2": "b"}
    launch_benchmark.run_bare_metal("/foo", "/bar", "/baz", test_env_vars, os_type=os_type)
    assert mock_popen.called
    args, kwargs = mock_popen.call_args

    # make sure that the start script is run
    assert "bash" == args[0][0]
    assert "start.sh" in args[0][1]

    # ensure env vars are set
    assert os.environ["TEST_ENV_VAR_1"] == test_env_vars["TEST_ENV_VAR_1"]
    assert os.environ["TEST_ENV_VAR_2"] == test_env_vars["TEST_ENV_VAR_2"]


@pytest.mark.skip
@pytest.mark.parametrize('launch_benchmark', [["--in-graph", test_input_graph]], indirect=True)
def test_launch_benchmark_tensorflow_serving_framework(launch_benchmark, mock_popen):
    """
    Tests that the launch script works for tensorflow serving framework
    """
    test_env_vars = {"TEST_ENV_VAR_1": "a", "TEST_ENV_VAR_2": "b", "MPI_NUM_PROCESSES": "None"}
    # Override framework and docker image.
    launch_benchmark.args.framework = test_tfserving_framework
    launch_benchmark.args.docker_image = None
    launch_benchmark.run_bare_metal("/foo", "/bar", "/baz", test_env_vars)
    assert mock_popen.called
    args, kwargs = mock_popen.call_args

    assert launch_benchmark.args.input_graph == test_input_graph
    assert launch_benchmark.args.framework == test_tfserving_framework

    # make sure that the start script is run
    assert "bash" == args[0][0]
    assert "start.sh" in args[0][1]

    # ensure env vars are set
    assert os.environ["TEST_ENV_VAR_1"] == test_env_vars["TEST_ENV_VAR_1"]
    assert os.environ["TEST_ENV_VAR_2"] == test_env_vars["TEST_ENV_VAR_2"]


def test_help(mock_platform_util, capsys):
    """ Tests `launch_benchmark.py --help` output and ensures there is no error """
    with mock_patch.object(sys, 'argv', ["launch_benchmark.py", "--help"]):
        with pytest.raises(SystemExit) as e:
            LaunchBenchmark(mock_platform_util)
        assert e.value.code == 0

        # get the stdout and check the output
        captured = capsys.readouterr()
        assert "usage: launch_benchmark.py [-h] " in captured.out

        # check for an arg that is only in launch_benchmark.py
        assert "--docker-image DOCKER_IMAGE" in captured.out

        # check for an arg that's in base_benchmark_util.py
        assert "-f FRAMEWORK, --framework FRAMEWORK" in captured.out

        # make sure there were no errors printed
        assert "error" not in captured.out.lower()


def test_launch_benchmark_custom_volume(launch_benchmark, mock_popen, platform_mock):
    """
    Verifies the docker run command includes custom volumes
    """
    custom_volumes = ["~:/foo1", "~:/foo2"]
    launch_benchmark.args.custom_volumes = custom_volumes
    platform_mock.return_value = platform_config.OS_TYPE
    launch_benchmark.main()
    assert mock_popen.called
    args, _ = mock_popen.call_args
    # convert the run command args to a string and then check for the custom volume mounts
    docker_run_cmd = " ".join(args[0])
    for custom_volume in custom_volumes:
        assert "--volume {}".format(custom_volume) in docker_run_cmd


@pytest.mark.parametrize("precision,expected_disable_tcmalloc", [["int8", "False"],
                                                                 ["fp32", "True"]])
def test_disable_tcmalloc(launch_benchmark, mock_popen,
                          platform_mock, precision, expected_disable_tcmalloc):
    launch_benchmark.args.precision = precision
    platform_mock.return_value = platform_config.OS_TYPE
    launch_benchmark.main()
    assert mock_popen.called
    args, _ = mock_popen.call_args
    # convert the run command args to a string and then check for the custom volume mounts
    docker_run_cmd = " ".join(args[0])
    assert "--env DISABLE_TCMALLOC=".format(expected_disable_tcmalloc) in docker_run_cmd


@pytest.mark.parametrize("numa_cores_per_instance_arg,socket_id_args,num_cores_arg,mpi_num_proc_arg,run_privileged",
                         [["4", -1, -1, None, True],
                          [None, -1, -1, None, False],
                          ["socket", -1, -1, None, True],
                          [None, 0, -1, None, True],
                          [None, 1, -1, None, True],
                          [None, -1, 8, None, True],
                          [None, -1, -1, 2, True]])
def test_launch_benchmark_docker_privileged(launch_benchmark, mock_popen, platform_mock,
                                            numa_cores_per_instance_arg, socket_id_args,
                                            num_cores_arg, mpi_num_proc_arg, run_privileged):
    """
    Verifies that docker only runs with --privileged when it needs to (if args that
    run multi-instance or numactl are used).
    """
    launch_benchmark.args.numa_cores_per_instance = numa_cores_per_instance_arg
    launch_benchmark.args.socket_id = socket_id_args
    launch_benchmark.args.num_cores = num_cores_arg
    launch_benchmark.args.mpi = mpi_num_proc_arg
    platform_mock.return_value = platform_config.OS_TYPE
    launch_benchmark.main()
    assert mock_popen.called
    args, _ = mock_popen.call_args
    # convert the run command args to a string and then check for the docker args
    docker_run_cmd = " ".join(args[0])
    assert ("--privileged" in docker_run_cmd) == run_privileged
