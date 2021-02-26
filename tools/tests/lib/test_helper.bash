#!/usr/bin/env bash

load "lib/utils"
load "lib/detik"

DETIK_CLIENT_NAME="kubectl"

# create namespace $USER if not exist
kubectl get ns -oname | grep $USER || kubectl create ns $USER
$DETIK_CLIENT_NAME config set-context $($DETIK_CLIENT_NAME config current-context) --namespace=$USER

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
# *** tests for generate-deployment ***