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

