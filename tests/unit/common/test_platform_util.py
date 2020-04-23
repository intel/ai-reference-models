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

import pytest
from mock import MagicMock

from benchmarks.common.platform_util import PlatformUtil
from test_utils import platform_config


def setup_mock_values(mock_platform, mock_os, mock_subprocess):
    platform_config.set_mock_system_type(mock_platform)
    platform_config.set_mock_os_access(mock_os)
    platform_config.set_mock_lscpu_subprocess_values(mock_subprocess)


@pytest.fixture
def os_mock(patch):
    return patch("os.access")


@pytest.fixture
def subprocess_mock(patch):
    return patch("subprocess.check_output")


@pytest.fixture
def platform_mock(patch):
    return patch("system_platform.system")


def test_platform_util_lscpu_parsing(platform_mock, subprocess_mock, os_mock):
    """
    Verifies that platform_utils gives us the proper values that we expect
    based on the lscpu_output string provided.
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    os_mock.return_value = True
    subprocess_mock.side_effect = [platform_config.LSCPU_PATH,
                                   platform_config.LSCPU_OUTPUT]
    platform_util = PlatformUtil(MagicMock(verbose=True))
    assert platform_util.num_cpu_sockets == 2
    assert platform_util.num_cores_per_socket == 28
    assert platform_util.num_threads_per_core == 2
    assert platform_util.num_logical_cpus == 112
    assert platform_util.num_numa_nodes == 2


def test_platform_util_unsupported_os(platform_mock, subprocess_mock, os_mock):
    """
    Verifies that platform_utils gives us the proper values that we expect
    based on the lscpu_output string provided.
    """
    os_mock.return_value = True
    subprocess_mock.side_effect = [platform_config.LSCPU_PATH,
                                   platform_config.LSCPU_OUTPUT]
    # Mac is not supported yet
    platform_mock.return_value = "Mac"
    with pytest.raises(NotImplementedError) as e:
        PlatformUtil(MagicMock(verbose=True))
    assert "Mac Support not yet implemented" in str(e)
    # Windows is not supported yet
    platform_mock.return_value = "Windows"
    with pytest.raises(NotImplementedError) as e:
        PlatformUtil(MagicMock(verbose=False))
    assert "Windows Support not yet implemented" in str(e)
