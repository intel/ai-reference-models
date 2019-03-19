#
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


def set_env_var(env_var, value, overwrite_existing=False):
    """
    Sets the specified environment variable.

    If overwrite_existing is False, it will only set the new env var value
    if the environment variable is not already set.

    If overwrite_existing is True, the environment variable will always be
    set to the specified value.
    """
    if overwrite_existing or not os.environ.get(env_var):
        os.environ[env_var] = str(value)


class BaseModelInitializer(object):

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        if not platform_util:
            raise ValueError("Did not find any platform info.")

    def run_command(self, cmd):
        """
        Prints debug messages when verbose is enabled, and then runs the
        specified command.
        """
        if self.args.verbose:
            print("Received these standard args: {}".format(self.args))
            print("Received these custom args: {}".format(self.custom_args))
            print("Current directory: {}".format(os.getcwd()))
            print("Running: {}".format(str(cmd)))

        os.system(cmd)

    def get_numactl_command(self, socket_id):
        """
        Returns the numactl command with --cpunodebind and --membind set to the
        specified socket_id.  If socket_id is set to -1 (undefined) then an
        empty string is returned.
        """
        return "" if socket_id == -1 else \
            "numactl --cpunodebind={0} --membind={0} ".format(
                str(socket_id))

    def add_args_to_command(self, command, arg_list):
        """
        Add args that are specified in the arg list to the command.  batch_size
        is a special case, where it's not added if it's set to -1 (undefined).
        Returns the command string with args.
        """
        for arg in vars(self.args):
            arg_value = getattr(self.args, arg)
            if arg == "batch_size" and arg_value == -1:
                continue
            if arg_value and (arg in arg_list):
                command = "{cmd} --{param}={value}".format(
                    cmd=command, param=arg, value=arg_value)
        return command

    def set_num_inter_intra_threads(self, num_inter_threads=None, num_intra_threads=None):
        """
        Sets default values for self.args.num_inter_threads and
        self.args.num_intra_threads, only if they are not already set.

        If num_inter_threads and/or num_intra_threads are specified, then those
        are the values that will be used. Otherwise, if they are None, then the
        following criteria applies:

        If a single socket is being used:
         * num_inter_threads = 1
         * num_intra_threads = The number of cores on a single socket, or
           self.args.num_cores if a specific number of cores was defined.

        If all sockets are being used:
         * num_inter_threads = The number of sockets
         * num_intra_threads = The total number of cores across all sockets, or
           self.args.num_cores if a specific number of cores was defined.
        """
        # if num_inter_threads is specified, use that value as long as the arg isn't set
        if num_inter_threads and not self.args.num_inter_threads:
            self.args.num_inter_threads = num_inter_threads

        # if num_intra_threads is specified, use that value as long as the arg isn't set
        if num_intra_threads and not self.args.num_intra_threads:
            self.args.num_intra_threads = num_intra_threads

        if self.args.socket_id != -1:
            if not self.args.num_inter_threads:
                self.args.num_inter_threads = 1
            if not self.args.num_intra_threads:
                self.args.num_intra_threads = \
                    self.platform_util.num_cores_per_socket \
                    if self.args.num_cores == -1 else self.args.num_cores
        else:
            if not self.args.num_inter_threads:
                self.args.num_inter_threads = self.platform_util.num_cpu_sockets
            if not self.args.num_intra_threads:
                if self.args.num_cores == -1:
                    self.args.num_intra_threads = \
                        int(self.platform_util.num_cores_per_socket *
                            self.platform_util.num_cpu_sockets)
                else:
                    self.args.num_intra_threads = self.args.num_cores

        if self.args.verbose:
            print("num_inter_threads: {}\nnum_intra_threads: {}".format(
                self.args.num_inter_threads, self.args.num_intra_threads))

    def set_kmp_vars(self, kmp_settings="1", kmp_blocktime="1", kmp_affinity="granularity=fine,verbose,compact,1,0"):
        """
        Sets KMP_* environment variables to the specified value, if the environment variable has not already been set.
        The default values for this function's args are the most common values that we have seen in the model zoo.
        """
        if kmp_settings:
            set_env_var("KMP_SETTINGS", kmp_settings)
        if kmp_blocktime:
            set_env_var("KMP_BLOCKTIME", kmp_blocktime)
        if kmp_affinity:
            set_env_var("KMP_AFFINITY", kmp_affinity)
