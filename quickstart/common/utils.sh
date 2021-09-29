#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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


_command()
{
  local _args="$@" _count _pattern=' -- ' _tmp
  _tmp="${_args//$_pattern}"
  # check for duplicate ' -- ' and remove the latter
  _count=$(((${#_args} - ${#_tmp}) / ${#_pattern}))
  if (( $_count > 1 )); then
    _args="${_args%${_pattern}*}"' '"${_args##*${_pattern}}"
  fi
  if [[ ${_args[@]} =~ --dry-run ]]; then
    echo "${_args[@]}"
  fi
  echo $@
  echo ""
  eval $@
}

_ht_status_spr()
{
  # Intel Optimizations specific Envs for TensorFlow SPR
  # HT on/off with KMP_AFFINITY:
  # HT - on (use KMP_AFFINITY=granularity=fine,verbose,compact,1,0)
  # HT - off (use KMP_AFFINITY=granularity=fine,verbose,compact,)

  HT_STATUS=$(lscpu |grep 'Thread' |sed 's/[^0-9]//g')
  if [[ ${HT_STATUS} == "1" ]] ; then
    export KMP_AFFINITY='granularity=fine,verbose,compact'
  elif [[ ${HT_STATUS} == "2" ]] ; then
    export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
  fi
  echo ""
  echo "Setting env var KMP_AFFINITY=${KMP_AFFINITY}"
  echo ""
}

_get_numa_cores_lists()
{
  cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
  sockets=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
  number_of_cores=$(($cores_per_socket * $sockets))
  echo "number of physical cores: ${number_of_cores}"
  numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
  echo "number of NUMA nodes: ${numa_nodes_num}"
  cores_per_node=$((number_of_cores/numa_nodes_num))
  cores_arr=()
  for ((i=0;i<${numa_nodes_num};i++)); do
    node_cores=$(numactl -H |grep 'node '$i' cpus:' |sed 's/.*node '$i' cpus: *//')
    cores_arr[$i]=${node_cores// /,}
  done
}
