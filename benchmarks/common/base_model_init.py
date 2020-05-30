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

#

import glob
import json
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

        # Set default values for TCMalloc and convert string value to a boolean
        if self.args.disable_tcmalloc is None:
            # Set to False for int8 and True for other precisions
            self.args.disable_tcmalloc = self.args.precision != "int8"
        elif isinstance(self.args.disable_tcmalloc, str):
            self.args.disable_tcmalloc = self.args.disable_tcmalloc == "True"

        # Ensure that we are using the proper version of python to run the benchmarking script
        self.python_exe = os.environ["PYTHON_EXE"]

        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # Invoke mpirun if mpi_num_processes env is not None
        if os.environ["MPI_NUM_PROCESSES"] != "None":
            if os.environ["MPI_NUM_PROCESSES_PER_SOCKET"] == "1":
              # Map by socket using OpenMPI by default (PPS=1).
              self.python_exe = "mpirun --allow-run-as-root -n " + os.environ["MPI_NUM_PROCESSES"] + " --map-by socket " + self.python_exe
            else:
              # number of processes per socket (pps)
              pps = int(os.environ["MPI_NUM_PROCESSES_PER_SOCKET"])
              split_a_socket = str(platform_util.num_cores_per_socket // pps)
              # Launch pps MPI processes over one socket
              self.python_exe = "mpirun --allow-run-as-root -n " + os.environ["MPI_NUM_PROCESSES"] + " --map-by ppr:" + str(pps) + ":socket:pe=" + split_a_socket + " --cpus-per-proc " + split_a_socket + " " + self.python_exe

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

    def get_command_prefix(self, socket_id, numactl=True):
        """
        Returns the command prefix with:
         - LD_PRELOAD for int8 models (if tcmalloc is not disabled)
         - The numactl command with --cpunodebind and --membind set to the specified socket_id (if numactl=True)

        Should be used only for single instance.
        """
        command = ""

        if not self.args.disable_tcmalloc:
            # Try to find the TCMalloc library file
            matches = glob.glob("/usr/lib/libtcmalloc.so*")

            if len(matches) == 0:
                matches = glob.glob("/usr/lib64/libtcmalloc.so*")

            if len(matches) > 0:
                command += "LD_PRELOAD={} ".format(matches[0])
            else:
                # Unable to find the TCMalloc library file
                print("Warning: Unable to find the TCMalloc library file (libtcmalloc.so) in /usr/lib or /usr/lib64, "
                      "so the LD_PRELOAD environment variable will not be set.")

        if socket_id != -1 and numactl:
            command += "numactl --cpunodebind={0} --membind={0} ".format(str(socket_id))

        return command

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
         * in case MPI_NUM_PROCESSES is used
           * num_inter_threads = 1
           * num_intra_threads = the number of cores on a single socket minus 2
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
                if os.environ["MPI_NUM_PROCESSES"] != "None":
                  self.args.num_inter_threads = 1
            if not self.args.num_intra_threads:
                if self.args.num_cores == -1:
                    self.args.num_intra_threads = \
                        int(self.platform_util.num_cores_per_socket *
                            self.platform_util.num_cpu_sockets)
                    if os.environ["MPI_NUM_PROCESSES"] != "None":
                      self.args.num_intra_threads = \
                             self.platform_util.num_cores_per_socket - 2
                else:
                    self.args.num_intra_threads = self.args.num_cores

        if self.args.verbose:
            print("num_inter_threads: {}\nnum_intra_threads: {}".format(
                self.args.num_inter_threads, self.args.num_intra_threads))

    def set_kmp_vars(self, config_file_path, kmp_settings=None, kmp_blocktime=None, kmp_affinity=None):
        """
        Sets KMP_* environment variables to the specified value, if the environment variable has not already been set.
        The default values in the json file are the best known settings for the model.
        """
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as config:
                config_object = json.load(config)

            # First sets default from config file
            for param in config_object.keys():
                for env in config_object[param].keys():
                    set_env_var(env, config_object[param][env])

        else:
            print("Warning: File {} does not exist and \
            cannot be used to set KMP environment variables".format(config_file_path))

        # Override user provided envs
        if kmp_settings:
            set_env_var("KMP_SETTINGS", kmp_settings, overwrite_existing=True)
        if kmp_blocktime:
            set_env_var("KMP_BLOCKTIME", kmp_blocktime, overwrite_existing=True)
        if kmp_affinity:
            set_env_var("KMP_AFFINITY", kmp_affinity, overwrite_existing=True)
