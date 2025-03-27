#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
import os
import tempfile
from argparse import ArgumentTypeError

import pytest
from mock import MagicMock
from test_utils import platform_config

from common.utils.validators import (
    check_for_link,
    check_no_spaces,
    check_positive_number,
    check_positive_number_or_equal_to_negative_one,
    check_valid_filename,
    check_valid_folder,
    check_valid_file_or_dir,
    check_volume_mount,
    check_shm_size,
    check_num_cores_per_instance,
)


@pytest.fixture()
def mock_link(patch):
    return patch("common.utils.validators.check_for_link")


@pytest.fixture()
def mock_exists(patch):
    return patch("os.path.exists")


@pytest.fixture()
def mock_isfile(patch):
    return patch("os.path.isfile", MagicMock(return_value=True))


def test_check_for_link_file():
    """
    Tests check_for_link to ensure that sym links and hard links
    are not allowed. Creates a symlink and hard link of a temporary file and
    verifies that things fail with the appropriate error message
    """

    with tempfile.NamedTemporaryFile() as temp_file:
        # directory where the temp file is located
        parent_dir = os.path.dirname(temp_file.name)

        # create sym link to the temp file
        symlink_file = os.path.join(parent_dir, "temp_symlink_file")
        if os.path.exists(symlink_file):
            os.remove(symlink_file)
        os.symlink(temp_file.name, symlink_file)

        # create hard link to the temp file
        hardlink_file = os.path.join(parent_dir, "temp_hardlink_file")
        if os.path.exists(hardlink_file):
            os.remove(hardlink_file)
        os.link(temp_file.name, hardlink_file)

        try:
            # Test that hard link errors
            with pytest.raises(ArgumentTypeError) as e:
                check_for_link(hardlink_file)
            assert "cannot be a link" in str(e)

            # Test that sym link errors
            with pytest.raises(ArgumentTypeError) as e:
                check_for_link(symlink_file)
            assert "cannot be a link" in str(e)
        finally:
            if os.path.exists(symlink_file):
                os.remove(symlink_file)
            if os.path.exists(hardlink_file):
                os.remove(hardlink_file)


def test_check_for_link_folder():
    """
    Tests creating a directory path and ensuring that symlinks are not
    allowed. Creates a symlink of a temporary directory and verifies that things fail with proper error msg
    """
    # create temp directory
    temp_dir = tempfile.mkdtemp()
    parent_dir = os.path.dirname(temp_dir)

    # create sym link to the temp directory
    symlink_dir = os.path.join(parent_dir, "temp_symlink_dir")
    if os.path.exists(symlink_dir):
        os.remove(symlink_dir)
    os.symlink(temp_dir, symlink_dir)

    try:
        with pytest.raises(ArgumentTypeError) as e:
            check_for_link(symlink_dir)
        assert "cannot be a link" in str(e)
    finally:
        if os.path.exists(symlink_dir):
            os.remove(symlink_dir)
        os.rmdir(temp_dir)


def test_check_no_spaces():
    with pytest.raises(ArgumentTypeError):
        check_no_spaces("foo bar")


def test_check_positive_number():
    with pytest.raises(ArgumentTypeError):
        check_positive_number(-1)


def test_check_positive_number_or_equal_to_negative_one():
    with pytest.raises(ArgumentTypeError):
        check_positive_number_or_equal_to_negative_one(-2)


def test_check_valid_filename(mock_link, mock_isfile):
    with tempfile.NamedTemporaryFile() as temp_file:
        assert check_valid_filename(temp_file) == temp_file


def test_check_valid_filename_bad():
    with pytest.raises(ArgumentTypeError):
        check_valid_filename("3245jlnsdfnsfd234ofds")


def test_check_valid_folder(mock_link):
    temp_folder = tempfile.mkdtemp()
    assert check_valid_folder(temp_folder) == temp_folder


def test_check_valid_folder_bad():
    with pytest.raises(ArgumentTypeError):
        check_valid_folder("3245jlnsdfnsfd234ofds")


def test_check_valid_file_or_dir(mock_link, mock_exists):
    with tempfile.NamedTemporaryFile() as temp_file:
        assert check_valid_file_or_dir(temp_file) == temp_file


def test_check_valid_file_or_dir_bad():
    with pytest.raises(ArgumentTypeError):
        check_valid_file_or_dir("3245jlnsdfnsfd234ofds")


def test_check_invalid_shm_size():
    with pytest.raises(ArgumentTypeError):
        check_shm_size("-g123ff")


def test_check_valid_shm_size():
    assert check_shm_size("500g") == "500g"
    assert check_shm_size("64m") == "64m"
    assert check_shm_size("1024k") == "1024k"


@pytest.mark.parametrize("volume_mount_str", ["foo", "foo:foo:foo:foo", "foo,foo"])
def test_bad_volume_mount_strings(volume_mount_str):
    with pytest.raises(ArgumentTypeError):
        check_volume_mount(volume_mount_str)


@pytest.mark.skipif(
    platform_config.OS_TYPE == "Windows", reason="Windows supports baremetal only"
)
def test_valid_volume_mount():
    # create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # test string that mounts local directory with mount path
        volume_mount = temp_dir + ":/mount_path"
        check_volume_mount(volume_mount)

        # test string that mounts local directory with mount path and specifies read only
        volume_mount = temp_dir + ":/mount_path:ro"
        check_volume_mount(volume_mount)
    finally:
        os.rmdir(temp_dir)


@pytest.mark.parametrize("test_str", ["socket", "Socket", "SOCKET", "4", "8"])
def test_valid_num_cores_per_instance(test_str):
    assert check_num_cores_per_instance(test_str) == test_str.lower()


@pytest.mark.parametrize("test_str", ["foo", "0", "-1"])
def test_invalid_num_cores_per_instance(test_str):
    with pytest.raises(ArgumentTypeError):
        check_num_cores_per_instance(test_str)
