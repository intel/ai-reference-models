#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

# This script contains utility functions for working with TensorFlow's tf-tools
# environment, which we use for constructing Dockerfiles out of partials and 
# building images. It allows for checking if a tf-tools container already exists
# and building the tf-tools container if necessary.

_onexit()
{
  if (( $# > 0 )) && [[ -n $1 ]] && (( $1 != 0 )); then
    echo "Error $1 occurred on $2"
  fi
}

_echo_command()
{
  if ! [[ $@ =~ --quiet ]] && ! [[ $@ =~ -q ]]; then
    echo $@
  fi
  if ! [[ $@ =~ --dry-run ]]; then
    eval $@
  fi
}

_check-for-tf-tools()
{
  local _quiet=''
  if (( $# > 0 )); then
    case "$1" in 
      -q|--quiet)
         _quiet='true'
         ;;
      *)
         ;;
    esac
  fi
  test -z $_quiet && echo '> Checking for tf-tools:latest container' >&2
  docker inspect --format '{{index .RepoTags 0 }}' tf-tools:latest 2>/dev/null 1>/dev/null
}

_addText()
{
   local file="$1" line="$2" newText="$3"
   sed -i "" -e "/^$line$/a"$'\\\n'"$newText"$'\n' "$file"
}

_build-tf-tools()
{
  local _dir _quiet='' _proxy_build
  while [[ "$#" -gt "0" && $1 =~ ^- ]]; do
    case "$1" in 
      -q|--quiet)
         _quiet=$1
         shift
         ;;
      *)
         echo 'unknown argument '$1
         exit 1
         ;;
    esac
  done
  _dir=$1
  if ! [[ -d $_dir ]]; then
    echo $_dir' is not a directory'
    exit 1
  fi
  if [[ -n $http_proxy && -n $https_proxy && -n $no_proxy ]]; then
    _proxy_build=' --build-arg HTTPS_PROXY='$https_proxy' --build-arg HTTP_PROXY='$http_proxy' --build-arg NO_PROXY='$no_proxy
  fi
  test -z $_quiet && echo '> Building tf-tools docker image' >&2
  cat <<DOCKERIGNORE >.dockerignore
*
!bashrc
DOCKERIGNORE
  pushd $_dir 2>/dev/null 1>/dev/null
  docker build $_quiet $_proxy_build --tag tf-tools:latest -f tools.Dockerfile . && echo '> Build successful' >&2
  rm -f .dockerignore
  popd 2>/dev/null 1>/dev/null
}

_get-proxy-env-vars()
{
  local _proxy_run=''
  if [[ -n $http_proxy && -n $https_proxy && -n $no_proxy ]]; then
    _proxy_run=' -e HTTPS_PROXY='$https_proxy' -e HTTP_PROXY='$http_proxy' -e NO_PROXY='$no_proxy
  fi
  echo "$_proxy_run"
}
