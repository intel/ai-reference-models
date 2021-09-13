
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
