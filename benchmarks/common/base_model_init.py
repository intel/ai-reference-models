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
import re
import sys
import time


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

        # use case: bare-metal with openmpi, horovod and multi-node
        if os.environ["MPI_HOSTNAMES"] != "None" and ("DOCKER" not in os.environ or os.environ["DOCKER"] == "False"):
            if os.environ["MPI_NUM_PROCESSES"] != "None":
                try:
                    # slots per host calculation using MPI_NUM_PROCESSES and number of hosts
                    host_names = os.environ["MPI_HOSTNAMES"]
                    number_of_hosts = len(host_names.split(','))
                    slots_per_host = int(int(os.environ["MPI_NUM_PROCESSES"]) / number_of_hosts)
                    host_names = ",".join([host + ":" + str(slots_per_host) for host in host_names.split(',')])
                    # see the [examples](https://horovod.readthedocs.io/en/latest/mpirun.html) for the mca flags
                    self.python_exe = "mpirun " + " -x LD_LIBRARY_PATH " + " -x PYTHONPATH " \
                        + " --allow-run-as-root -n " + os.environ["MPI_NUM_PROCESSES"] + " -H " + host_names \
                        + " -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude " \
                          "lo,docker0 --bind-to none --map-by slot " \
                        + self.python_exe
                except Exception as exception:
                    raise ValueError("Caught exception calculating slots per host {}".format(str(exception)))
            else:
                raise ValueError("MPI_NUM_PROCESSES is required for MPI_HOSTNAMES and will be split evenly across the "
                                 "hosts.")
        # use case: docker with openmpi, single-node, multi-instance
        elif os.environ["MPI_NUM_PROCESSES"] != "None":
            if os.environ["MPI_NUM_PROCESSES_PER_SOCKET"] == "1":
                # Map by socket using OpenMPI by default (PPS=1).
                self.python_exe = "mpirun --allow-run-as-root -n " + os.environ["MPI_NUM_PROCESSES"] \
                    + " --map-by socket " + self.python_exe
            else:
                # number of processes per socket (pps)
                pps = int(os.environ["MPI_NUM_PROCESSES_PER_SOCKET"])
                split_a_socket = str(platform_util.num_cores_per_socket // pps)
                # Launch pps MPI processes over one socket
                self.python_exe = "mpirun --allow-run-as-root -n " + os.environ["MPI_NUM_PROCESSES"] \
                    + " --map-by ppr:" + str(pps) + ":socket:pe=" + split_a_socket + " --cpus-per-proc " \
                    + split_a_socket + " " + self.python_exe

    def run_command(self, cmd, replace_unique_output_dir=None):
        """
        Prints debug messages when verbose is enabled, and then runs the
        specified command.

        If the replace_unique_output_dir arg is set, multi-instance runs will
        swap out that path for a path with the instance number in the folder name
        so that each instance uses a unique output folder.
        """
        if self.args.verbose:
            print("Received these standard args: {}".format(self.args))
            print("Received these custom args: {}".format(self.custom_args))
            print("Current directory: {}".format(os.getcwd()))

        if self.args.numa_cores_per_instance:
            num_numas = self.platform_util.num_numa_nodes

            if not num_numas:
                print("Warning: Unable to run multiple instances using numactl, "
                      "because no numa nodes were found.")
            elif not self.platform_util.cpu_core_list:
                print("Warning: Unable to run multiple instances using numactl, "
                      "the list of cpu nodes could not be retrieved. Please ensure "
                      "that your system has numa nodes and numactl is installed.")
            else:
                self.run_numactl_multi_instance(
                    cmd, replace_unique_output_dir=replace_unique_output_dir)
        else:
            if self.args.verbose:
                print("Running: {}".format(str(cmd)))

            os.system(cmd)

    def group_cores(self, cpu_cores_list, cores_per_instance):
        """
        Group cores based on the number of cores we want per instance.
        Returns a 2D array with the list of cores for each instance.
        """
        list_of_groups = zip(*(iter(cpu_cores_list),) * cores_per_instance)
        end_list = [list(i) for i in list_of_groups]
        count = len(cpu_cores_list) % cores_per_instance
        end_list.append(cpu_cores_list[-count:]) if count != 0 else end_list
        return end_list

    def run_numactl_multi_instance(self, cmd, replace_unique_output_dir=None):
        """
        Generates a series of commands that call the specified cmd with multiple
        instances, where each instance uses the a specified number of cores. The
        number of cores used per instance is specified by args.numa_cores_per_instance.

        The command for each instance uses numactl and the --physcpubind arg with
        the appropriate core list. Each instance writes output to it's own log file,
        and a combined log file is created after everything has executed.

        If the replace_unique_output_dir arg is set, multi-instance runs will
        swap out that path for a path with the instance number in the folder name
        so that each instance uses a unique output folder.
        """

        # Find LD_PRELOAD vars, remove them from the cmd, and save them to add on to the prefix
        ld_preload_strs = re.findall(r'\bLD_PRELOAD=\S*', cmd)
        ld_preload_prefix = ""
        for ld_preload_str in ld_preload_strs:
            cmd = cmd.replace(ld_preload_str, "")
            ld_preload_prefix += ld_preload_str + " "

        # Remove leading/trailing whitespace
        cmd = cmd.strip()

        if self.args.numa_cores_per_instance != "socket":
            # Get the cores list and group them according to the number of cores per instance
            cores_per_instance = int(self.args.numa_cores_per_instance)
            cpu_cores_list = self.platform_util.cpu_core_list

            if self.args.socket_id != -1:
                # If it's specified to just use a single socket, then only use the cores from that socket
                if len(cpu_cores_list) > self.args.socket_id:
                    cpu_cores_list = cpu_cores_list[self.args.socket_id]
                else:
                    raise ValueError("Error while trying to get the core list for socket {0}. "
                                     "The core list does not have cores for socket {0}.\n "
                                     "Core list: {1}\n".format(self.args.socket_id, str(cpu_cores_list)))
            else:
                # Using cores from all sockets
                combined_core_list = []
                for socket_cores in cpu_cores_list:
                    combined_core_list += socket_cores
                cpu_cores_list = combined_core_list

            instance_cores_list = self.group_cores(cpu_cores_list, cores_per_instance)
        else:
            instance_cores_list = []
            cores_per_instance = "socket"
            # Cores should be grouped based on the cores for each socket
            if self.args.socket_id != -1:
                # Only using cores from one socket
                instance_cores_list[0] = self.platform_util.cpu_core_list[self.args.socket_id]
            else:
                # Get the cores for each socket
                instance_cores_list = self.platform_util.cpu_core_list

        # Setup the log file name with the model name, precision, mode, batch size (if there is one),
        # number of cores per instance. An extra {} is intentionally left in the log_filename_format
        # string, because this value is filled in with the instance number later on.
        batch_size = ""
        if self.args.batch_size and self.args.batch_size > 0:
            batch_size = "bs{}_".format(self.args.batch_size)
        log_filename_format = os.path.join(
            self.args.output_dir, "{}_{}_{}_{}cores{}_".format(
                self.args.model_name, self.args.precision, self.args.mode, batch_size, cores_per_instance))
        log_filename_format += "{}.log"
        instance_logfiles = []

        # Loop through each instance and add that instance's command to a string
        multi_instance_command = ""
        for instance_num, core_list in enumerate(instance_cores_list):
            if cores_per_instance != "socket" and len(core_list) < int(cores_per_instance):
                print("NOTE: Skipping remainder of {} cores for instance {}"
                      .format(len(core_list), instance_num))
                continue

            if len(core_list) == 0:
                continue

            prefix = ("{0}OMP_NUM_THREADS={1} "
                      "numactl --localalloc --physcpubind={2}").format(
                ld_preload_prefix, len(core_list), ",".join(core_list))
            instance_logfile = log_filename_format.format("instance" + str(instance_num))

            unique_command = cmd
            if replace_unique_output_dir:
                # Swap out the output dir for a unique dir
                unique_dir = os.path.join(replace_unique_output_dir,
                                          "instance_{}".format(instance_num))
                unique_command = unique_command.replace(replace_unique_output_dir, unique_dir)

            instance_command = "{} {}".format(prefix, unique_command)
            multi_instance_command += "{} >> {} 2>&1 & \\\n".format(
                instance_command, instance_logfile)
            instance_logfiles.append(instance_logfile)

            # write the command to the instance's log file
            with open(instance_logfile, "w") as log:
                log.write(instance_command)
                log.write("\n\n")

        multi_instance_command += "wait"

        # Run the multi-instance command
        print("\nMulti-instance run:\n" + multi_instance_command)
        sys.stdout.flush()
        os.system(multi_instance_command)

        # Wait to ensure that log files have been written
        max_retries = 20
        retry_counter = 0
        while retry_counter < max_retries:
            if all([os.path.exists(log) for log in instance_logfiles]):
                break
            retry_counter += 1
            if retry_counter >= max_retries:
                print("Warning: Log files for all instances were not found after "
                      "rechecking and waiting for {} seconds. The combined log file "
                      "may not have output from all instances.".format(retry_counter))
                break
            time.sleep(1)

        # Generate the combined log file
        all_instance_log = log_filename_format.format("all_instances")
        os.environ["LOG_FILENAME"] = os.path.basename(all_instance_log)
        with open(all_instance_log, mode="w") as combined_file:
            for instance_logfile in instance_logfiles:
                if not os.path.exists(instance_logfile):
                    print("Skipping {} when generating the combined log file, because "
                          "it doesn't exist".format(os.path.basename(instance_logfile)))
                    continue

                with open(instance_logfile) as individual_file:
                    for line in individual_file:
                        combined_file.write(line)

        # Print out lists of log files
        print("\nThe following log files were saved to the output directory:")
        print("\n".join([os.path.basename(log_path) for log_path in instance_logfiles
                         if os.path.exists(log_path)]))
        if os.path.exists(all_instance_log):
            print("\nA combined log file was saved to the output directory:\n"
                  "{}\n".format(os.path.basename(all_instance_log)))

    def get_command_prefix(self, socket_id, numactl=True):
        """
        Returns the command prefix with:
         - LD_PRELOAD for int8 models (if tcmalloc is not disabled)
         - The numactl command with --cpunodebind and --membind set to the specified socket_id (if numactl=True)

        Should be used only for single instance.
        """
        command = ""
        ld_preload = ""

        if not self.args.disable_tcmalloc:
            # Try to find the TCMalloc library file
            matches = glob.glob("/usr/lib/libtcmalloc.so*")

            if len(matches) == 0:
                matches = glob.glob("/usr/lib64/libtcmalloc.so*")

            if len(matches) == 0:
                matches = glob.glob("/usr/lib/*/libtcmalloc.so*")

            if len(matches) == 0:
                matches = glob.glob("/usr/lib64/*/libtcmalloc.so*")

            if len(matches) > 0:
                ld_preload += "LD_PRELOAD={} ".format(matches[0])
            else:
                # Unable to find the TCMalloc library file
                print("Warning: Unable to find the TCMalloc library file (libtcmalloc.so) in /usr/lib, /usr/lib64, "
                      "/usr/lib/*, or /usr/lib64/* so the LD_PRELOAD environment variable will not be set.")

        num_numas = self.platform_util.num_numa_nodes
        if num_numas and socket_id != -1 and numactl and not self.args.numa_cores_per_instance:
            if self.args.num_cores == -1:
                # Running on the whole socket
                command += "numactl --cpunodebind={0} --membind={0} ".format(
                    str(socket_id))
            else:
                # Running on specific number of cores
                first_physical_core = self.platform_util.cpuset_cpus[0][0]
                num_sockets = len(self.platform_util.cpuset_cpus.keys())
                num_cores_in_socket0 = len(self.platform_util.cpuset_cpus[0])
                for i in range(num_sockets):
                    if num_cores_in_socket0 != len(
                            self.platform_util.cpuset_cpus[i]):
                        raise ValueError(
                            "Error: Identifying logical core id assumes all sockets have same number of cores"
                        )
                first_logical_core = num_cores_in_socket0 * num_sockets
                if self.platform_util.num_threads_per_core == 1:
                    # HT is off
                    cpus_range = "{0}-{1}".format(
                        first_physical_core,
                        first_physical_core + self.args.num_cores - 1)
                else:
                    # HT is on.
                    cpus_range = "{0}-{1},{2}-{3}".format(
                        first_physical_core,
                        first_physical_core + self.args.num_cores - 1,
                        first_logical_core,
                        first_logical_core + self.args.num_cores - 1)
                command += "numactl -C{0} --membind=0 ".format(cpus_range)

        # Add LD_PRELOAD to the front of the command
        command = ld_preload + command

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

        If multiple instances are being used (specified with numa_cores_per_instance),
        then each instance should have:
        * num_inter_threads = 1
        * num_intra_threads = number of cores per instance

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

        if self.args.numa_cores_per_instance:
            # Set default num inter/intra threads if the user didn't provide specific values
            if self.args.numa_cores_per_instance == "socket":
                if self.args.socket_id != -1:
                    inter_threads = len(self.platform_util.cpu_core_list[self.args.socket_id])
                else:
                    # since we can only have one value for inter threads and the number of cores
                    # per socket can vary, if the cpuset is limited, get the lowest core count
                    # per socket and use that as the num inter threads
                    inter_threads = min([len(i) for i in self.platform_util.cpu_core_list if len(i) > 0])
            else:
                inter_threads = self.args.numa_cores_per_instance

            if not self.args.num_inter_threads:
                self.args.num_inter_threads = 1
            if not self.args.num_intra_threads:
                self.args.num_intra_threads = inter_threads
            if not self.args.data_num_inter_threads:
                self.args.data_num_inter_threads = 1
            if not self.args.data_num_intra_threads:
                self.args.data_num_intra_threads = inter_threads
        elif self.args.socket_id != -1:
            if not self.args.num_inter_threads:
                self.args.num_inter_threads = 1
            if not self.args.num_intra_threads:
                if self.args.num_cores != -1:
                    self.args.num_intra_threads = self.args.num_cores
                elif self.platform_util.cpuset_cpus and \
                        self.args.socket_id in self.platform_util.cpuset_cpus.keys():
                    self.args.num_intra_threads = len(self.platform_util.cpuset_cpus[self.args.socket_id])
                else:
                    self.args.num_intra_threads = self.platform_util.num_cores_per_socket
        else:
            if not self.args.num_inter_threads:
                self.args.num_inter_threads = self.platform_util.num_cpu_sockets
                if os.environ["MPI_NUM_PROCESSES"] != "None":
                    self.args.num_inter_threads = 1
            if not self.args.num_intra_threads:
                if self.args.num_cores == -1:
                    if self.platform_util.cpuset_cpus and len(self.platform_util.cpuset_cpus.keys()) > 0:
                        # Total up the number of cores in the cpuset
                        self.args.num_intra_threads = sum([len(self.platform_util.cpuset_cpus[socket_id])
                                                           for socket_id in self.platform_util.cpuset_cpus.keys()])
                    else:
                        self.args.num_intra_threads = \
                            int(self.platform_util.num_cores_per_socket *
                                self.platform_util.num_cpu_sockets)
                    if os.environ["MPI_NUM_PROCESSES"] != "None":
                        self.args.num_intra_threads = self.platform_util.num_cores_per_socket - 2
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
