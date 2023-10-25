#!/bin/bash

function gen_single_key()
{
    ALG_NAME=$1
    if [[ ! -f /etc/ssh/ssh_host_${ALG_NAME}_key ]]
    then
        ssh-keygen -q -N "" -t ${ALG_NAME} -f /etc/ssh/ssh_host_${ALG_NAME}_key
    fi
}


gen_single_key dsa
gen_single_key rsa
gen_single_key ecdsa
gen_single_key ed25519
