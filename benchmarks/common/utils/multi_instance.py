#
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

"""Multi instance utils module."""

from common.platform_util import CPUInfo


def buckets(array, bucket_size):
    """
    Split array into multiple arrays with specified size.
    :param array: array that will be splited
    :type array: List[Any]
    :param bucket_size_size: target arrays size
    :type bucket_size_size: int
    :return: list with parameters
    :rtype: List[List[Any]]
    """
    bucket_size_list = []
    for i in range(0, len(array), bucket_size):
        bucket_size_list.append(array[i : i + bucket_size])

    return bucket_size_list


class InferencePrefix:
    """Multi instance class."""

    def __init__(self, sockets=0, instances=0, cores_per_instance=0):
        """
        Initialize multi instance class.
        :param sockets: sockets used for execution, defaults to 0
        :type sockets: int, optional
        :param instances: number of instances, defaults to 0
        :type instances: int, optional
        :param cores_per_instance: number of cores that will be used by one instance, defaults to 0
        :type cores_per_instance: int, optional
        """
        self._cpu_information = CPUInfo()
        self._sockets = sockets
        self._instances = instances
        self._cores_per_instance = cores_per_instance

    @property
    def is_basic_configuration(self):
        """
        Check if workload is multi instance or should use core/memory binding.
        :return: True if basic configuration else False
        :rtype: bool
        """
        # Expected single instance parameters
        single_instance_params = self._platform_single_instance_args()

        # Current workload parameters
        default_cores_per_instance = (
            self._cpu_information.cores_per_socket * self.sockets
        )
        workload_params = {
            "cores_per_instance": self._cores_per_instance
            or default_cores_per_instance,
            "instance": self._instances if self._instances != 0 else 1,
            "sockets": self.sockets,
        }

        return single_instance_params == workload_params

    @property
    def sockets(self):
        """
        Return amount of sockets used for execution.
        :return: amount of sockets
        :rtype: int
        """
        if self._sockets == 0:
            sockets = self._cpu_information.sockets
        else:
            sockets = self._sockets
            if sockets > self._cpu_information.sockets:
                raise Exception(
                    "The specified number of sockets is greater "
                    "than the number of server available sockets."
                )

        return sockets

    @property
    def cores_per_socket(self):
        """
        Return amount of cores per socket used for execution.
        :raises Exception: Cores assigned to one instance > cores available on one socket
        :return: amount of cores
        :rtype: int
        """
        if self._cores_per_instance > 0:
            if self._cores_per_instance > self._cpu_information.cores_per_socket:
                raise Exception(
                    "Cores assigned to one instance is greater than amount of cores on one socket."
                )

            cores_per_socket = (
                self._cpu_information.cores_per_socket
                - self._cpu_information.cores_per_socket % self._cores_per_instance
            )
        else:
            cores_per_socket = self._cpu_information.cores_per_socket

        return cores_per_socket

    @property
    def cores(self):
        """
        Return amount of cores used for execution.
        :return: amount of cores used for execution
        :rtype: int
        """
        cores = self.cores_per_socket * self.sockets
        return cores

    @property
    def instances_per_socket(self):
        """
        Return number of instances.
        :return: number of instances
        :rtype: int
        """
        if self._instances > 0:
            if self._instances % self.sockets != 0:
                raise Exception(
                    "Instances could not be distributed equally between sockets. "
                    "Amount of instances should be divisible by socket amount. "
                    "{} % {} != 0".format(self._instances, self.sockets)
                )

            instances = int(self._instances / self.sockets)
        elif self._cores_per_instance > 0:
            instances = int(self.cores_per_socket / self._cores_per_instance)

        else:
            instances = 0

        return instances

    @property
    def instances(self):
        """
        Return total number of instances.
        :return: total number of instances
        :rtype: int
        """
        # Set number of instances to 1 if instances_per_socket == 0
        if self.is_basic_configuration:
            return 1
        else:
            return (self.instances_per_socket * self.sockets) or 1

    @property
    def cores_per_instance(self):
        """
        Return cores per instance.
        :return: amount of cores per instance
        :rtype: int
        """
        if not self.is_basic_configuration:
            if self._cores_per_instance > 0:
                if (
                    self._cores_per_instance * self.instances_per_socket
                    > self.cores_per_socket
                ):
                    raise Exception(
                        "Total cores used on one socket > cores available on one socket. "
                        "{} * {} > {}".format(
                            self._cores_per_instance,
                            self.instances_per_socket,
                            self.cores_per_socket,
                        )
                    )

                cores_per_instance = self._cores_per_instance
            else:
                instances_per_socket = self.instances_per_socket
                if self.cores_per_socket % instances_per_socket != 0:
                    raise Exception(
                        "Amount of cores per socket should be divisible by amount of instances per socket."
                    )

                cores_per_instance = self.cores_per_socket // instances_per_socket

        else:
            cores_per_instance = self._cpu_information.cores

        return int(cores_per_instance)  # type: ignore

    @property
    def sockets_per_instance(self):
        """
        Return amount of sockets per instance.
        :return: amount of sockets per instance
        :rtype: int
        """
        if self.is_basic_configuration:
            sockets = self._cpu_information.sockets
        else:
            sockets = 1

        return sockets

    @staticmethod
    def get_cores_range(cores, ht_cores, use_ht):
        """
        Return the range of cores.
        :param cores: number of cores
        :param ht_cores: number of cores with hyperthreading
        :param use_ht: defines if hyperthreading should be used
        :return: range of cores
        """
        if use_ht and ht_cores:
            cores_range = "{},{}".format(cores, ht_cores)
        else:
            cores_range = cores

        return cores_range

    def split_cores(self):
        """
        Return cores in instance buckets.
        :raises Exception: 1 instance on sockets > 1 not implemented
        :return: instance buckets
        :rtype: Dict[str, List[List[Dict[str, Any]]]]
        """
        membind_info = self._cpu_information.binding_information
        cores_per_instance = self.cores_per_instance
        if cores_per_instance == 0:
            raise Exception("1 instance on sockets > 1 not implemented.")

        bucketed_cores = {}
        for node_id in range(self.sockets):
            socket_cores = membind_info[node_id][: self.cores_per_socket]
            instance_buckets = buckets(socket_cores, cores_per_instance)
            bucketed_cores.update(
                {str(node_id): instance_buckets[0 : self.instances_per_socket]}
            )

        return bucketed_cores

    def generate_multi_instance_ranges(self, use_ht=False):
        """
        Create config for multi-instance execution.
        :param use_ht: defines if hyperthreading should be used
        :return: information about splitted cores
        """
        instance_binding = []
        split_cores = self.split_cores()
        for instance_buckets in split_cores.values():
            for instance_config in instance_buckets:
                if len(instance_config) == 1:
                    cores = instance_config[0].get("cpu_id")
                    ht_cores = instance_config[0].get("ht_cpu_id", None)
                else:
                    cores = "{first}-{last}".format(
                        first=instance_config[0].get("cpu_id"),
                        last=instance_config[-1].get("cpu_id"),
                    )

                    first_ht = instance_config[0].get("ht_cpu_id", None)
                    last_ht = instance_config[-1].get("ht_cpu_id", None)
                    if first_ht is None or last_ht is None:
                        ht_cores = None
                    else:
                        ht_cores = "{first}-{last}".format(first=first_ht, last=last_ht)

                cores_range = self.get_cores_range(cores, ht_cores, use_ht)
                instance_binding.append(
                    {
                        "cores_range": cores_range,
                        "socket_id": instance_config[0].get("socket_id"),
                    }
                )

        return instance_binding

    def generate_multi_instance_prefix(self, command, use_ht=False):
        """
        Add 'numactl' prefix for multi-instance execution.
        :param command: command that will be run using numactl
        :param use_ht: defines if hyperthreading should be used
        :return: array of commands if multi-instance else command
        """
        if self.is_basic_configuration:
            return [command]

        commands_array = []
        for instance in self.generate_multi_instance_ranges(use_ht):
            numa_cmd = [
                "numactl",
                "--membind={}".format(instance.get("socket_id")),
                "--physcpubind={}".format(instance.get("cores_range")),
            ]

            commands_array.append(numa_cmd + command)

        return commands_array

    def _platform_single_instance_args(self):
        """
        Return single instance parameters for current platform.
        :return: single instance parameters for current platform
        :rtype: Dict[str, int]
        """
        return {
            "cores_per_instance": self._cpu_information.cores,
            "instance": 1,
            "sockets": self._cpu_information.sockets,
        }
