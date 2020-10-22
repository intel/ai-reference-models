
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
  eval $@
}

