#
# Copyright (c) 2023 Intel Corporation
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
echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 > /proc/sys/kernel/numa_balancing
echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct


# Clean resources
echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo 1 > /proc/sys/vm/compact_memory; sleep 1
echo 3 > /proc/sys/vm/drop_caches; sleep 1
