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
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import print_function

import os
import pytest
import tempfile
from mock import patch

from launch_benchmark import LaunchBenchmark
from test_utils import platform_config


# Example args and output strings for testing mocks
test_model_name = "resnet50"
test_framework = "tensorflow"
test_mode = "inference"
test_precision = "fp32"
test_docker_image = "foo"
test_batch_size = "100"
test_num_cores = "1"
example_req_args = ["--model-name", test_model_name,
                    "--framework", test_framework,
                    "--mode", test_mode,
                    "--precision", test_precision,
                    "--docker-image", test_docker_image,
                    "--batch-size", test_batch_size,
                    "--num-cores", test_num_cores]


@pytest.fixture
def mock_platform_util(patch):
    return patch("base_benchmark_util.platform_util.PlatformUtil")


@pytest.fixture
def mock_os(patch):
    return patch("base_benchmark_util.platform_util.os")


@pytest.fixture
def mock_subprocess(patch):
    return patch("base_benchmark_util.platform_util.subprocess")


@pytest.fixture
def mock_system_platform(patch):
    return patch("base_benchmark_util.platform_util.system_platform")


def setup_mock_values(platform_mock, os_mock, subprocess_mock):
    platform_config.set_mock_system_type(platform_mock)
    platform_config.set_mock_os_access(os_mock)
    platform_config.set_mock_lscpu_subprocess_values(subprocess_mock)


def test_launch_benchmark_parse_args(mock_platform_util):
    """
    Verifies that that arg parsing gives us the expected results.
    """
    launch_benchmark = LaunchBenchmark()
    args, unknown_args = launch_benchmark.parse_args(example_req_args)
    assert args.model_name == test_model_name
    assert args.framework == test_framework
    assert args.mode == test_mode
    assert args.precision == test_precision
    assert args.docker_image == test_docker_image
    assert unknown_args == []


def test_launch_benchmark_parse_unknown_args(mock_platform_util):
    """
    Checks parsing of unknown args
    """
    launch_benchmark = LaunchBenchmark()
    test_args = example_req_args + ["--test", "foo"]
    args, unknown_args = launch_benchmark.parse_args(test_args)
    assert unknown_args == ["--test"]


def test_launch_benchmark_parse_bad_args(mock_platform_util):
    """
    Checks for a failure when no args are passed.
    """
    launch_benchmark = LaunchBenchmark()
    # arg parse should fail when no args are passed
    with pytest.raises(SystemExit):
        launch_benchmark.parse_args([])


def test_launch_benchmark_validate_args(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Tests that valid args pass arg validation without any errors.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    launch_benchmark.validate_args(args)


def test_launch_benchmark_validate_bad_framework(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that an unsupported framework name errors.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.framework = "foo"
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert "The specified framework is not supported" in str(e)


def test_launch_benchmark_validate_bad_checkpoint_dir(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that an invalid checkpoint path fails.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    bad_path = "/path/does/not_exist"
    args.checkpoint = bad_path
    with pytest.raises(IOError) as e:
        launch_benchmark.validate_args(args)
    assert "The checkpoint location {} does not exist".format(bad_path) \
        in str(e)

    # test with a file
    with tempfile.NamedTemporaryFile() as temp_file:
        args.checkpoint = temp_file.name
        with pytest.raises(IOError) as e:
            launch_benchmark.validate_args(args)
        assert "The checkpoint location {} is not a directory".format(
            temp_file.name) in str(e)


def test_launch_benchmark_validate_checkpoint_dir(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that a valid checkpoint path passes.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    temp_dir = tempfile.mkdtemp()
    args.checkpoint = temp_dir
    try:
        launch_benchmark.validate_args(args)
    finally:
        os.rmdir(temp_dir)


def test_launch_benchmark_validate_bad_in_graph(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that an invalid input graph path fails.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)

    # test with path that does not exist
    bad_path = "/path/does/not_exist"
    args.input_graph = bad_path
    with pytest.raises(IOError) as e:
        launch_benchmark.validate_args(args)
    assert "The input graph {} does not exist".format(bad_path) \
        in str(e)

    # test with path that is a directory
    temp_dir = tempfile.mkdtemp()
    args.input_graph = temp_dir
    try:
        with pytest.raises(IOError) as e:
            launch_benchmark.validate_args(args)
        assert "The input graph {} must be a file".format(temp_dir) \
            in str(e)
    finally:
        os.rmdir(temp_dir)


def test_launch_benchmark_validate_in_graph(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that a valid input graph path passes.
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    with tempfile.NamedTemporaryFile() as temp_file:
        args.input_graph = temp_file.name
        launch_benchmark.validate_args(args)


def test_launch_benchmark_validate_bad_batch_size(mock_platform_util):
    """
    Verifies that a bad batch size fails
    """
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.batch_size = 0
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert "The batch size 0 is not valid." in str(e)

    args.batch_size = -100
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert "The batch size -100 is not valid." in str(e)


def test_launch_benchmark_validate_num_cores(
        mock_system_platform, mock_os, mock_subprocess):
    """
    Verifies that a bad num cores fails
    """
    setup_mock_values(mock_system_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.num_cores = 0
    expected_error = ("Core number must be greater than 0 or -1. The default "
                      "value is -1 which means using all the cores in the "
                      "sockets")
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert expected_error in str(e)

    args.num_cores = -100
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert expected_error in str(e)


@patch("subprocess.Popen")
def test_launch_benchmark_validate_model(
        mock_popen, mock_platform_util):
    """
    Verifies that a valid model name passes validation and starts a docker
    container.
    """
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.model_name = "resnet50"
    launch_benchmark.run_docker_container(args)
    assert mock_popen.called
    args, kwargs = mock_popen.call_args
    assert "docker" == args[0][0]
    assert "run" == args[0][1]


def test_launch_benchmark_validate_bad_model(mock_platform_util):
    """
    Verifies that a bad model name fails
    """
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.model_name = "foo"
    with pytest.raises(ValueError) as e:
        launch_benchmark.run_docker_container(args)
    assert "No model was found for" in str(e)


def test_launch_benchmark_validate_empty_model(mock_platform_util):
    """
    Verifies that giving no model name fails
    """
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    args.model_name = ""
    with pytest.raises(ValueError) as e:
        launch_benchmark.validate_args(args)
    assert "The model name is not valid" in str(e)
