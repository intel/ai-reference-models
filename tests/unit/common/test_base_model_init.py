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

import os

from mock import patch

from benchmarks.common.base_model_init import BaseModelInitializer
from benchmarks.common.base_model_init import set_env_var
from benchmarks.launch_benchmark import LaunchBenchmark
from test_utils import platform_config


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


def setup_mock_values(mock_platform, mock_os, mock_subprocess):
    platform_config.set_mock_system_type(mock_platform)
    platform_config.set_mock_os_access(mock_os)
    platform_config.set_mock_lscpu_subprocess_values(mock_subprocess)


@patch("common.platform_util.os")
@patch("common.platform_util.system_platform")
@patch("common.platform_util.subprocess")
@patch("os.system")
def test_base_model_initializer(
        mock_system, mock_subprocess, mock_platform, mock_os):
    setup_mock_values(mock_platform, mock_os, mock_subprocess)
    launch_benchmark = LaunchBenchmark()
    args, _ = launch_benchmark.parse_args(example_req_args)
    test_run_command = "python foo.py"

    # Setup base model init with test settings
    base_model_init = BaseModelInitializer()
    base_model_init.args = args
    base_model_init.args.verbose = True
    base_model_init.args.model_name = test_model_name
    base_model_init.custom_args = []

    # call run_command and then check the output
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
