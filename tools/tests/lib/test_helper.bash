#!/usr/bin/env bash

load "lib/utils"
load "lib/detik"

DETIK_CLIENT_NAME="kubectl"

# create namespace $USER if not exist
user_name="$USER"
echo "-------------------------------------------"
if [[ $user_name =~ _ ]]; then
  user_name="${user_name//_/-}"
  echo " "
  echo "*** Cannot create namespace for USERNAME $USER contains '_'"
  echo "*** Tests will use USERNAME $user_name instead ..."
fi

(kubectl get ns -oname | grep $user_name || kubectl create ns $user_name) && $DETIK_CLIENT_NAME config set-context $($DETIK_CLIENT_NAME config current-context) --namespace=$user_name
echo " "

last_modified ()
{
    local _os=$(uname -s);
    case "$_os" in
        Darwin)
            echo $(( $(date +%s) - $(stat -f%c $1) ))
        ;;
        Linux)
            expr $(date +%s) - $(stat -c %Y $1)
        ;;
    esac
}

launcher_logs()
{
  local _launcher_pod=$($DETIK_CLIENT_NAME get pods --no-headers -oname | grep launcher | sed 's^pod/^^')
  if [[ -n $_launcher_pod ]]; then
    $DETIK_CLIENT_NAME logs $_launcher_pod | tail -5
  fi
}
# *** tests for generate-deployment ***