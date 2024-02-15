#!/bin/bash

# The working dir in the container is /host
source "${HOST_PATH}/${SCRIPT_ROOT}/shared_vars.sh"

#install the collated requirements
pip install -r "${HOST_PATH}/${SCRIPT_ROOT}/${COLLATED_REQUIREMENTS_FILENAME}"

check_error $? "Error installing collated requirements."

pip freeze > "${HOST_PATH}/${SCRIPT_ROOT}/${FROZEN_COLLATED_REQUIREMENTS_FILENAME}"
