HOST_PATH="/host"
COLLATED_REQUIREMENTS="requirements_collated"
COLLATED_REQUIREMENTS_FILENAME="${COLLATED_REQUIREMENTS}.txt"
FROZEN_COLLATED_REQUIREMENTS_FILENAME="${COLLATED_REQUIREMENTS}.frozen.txt"
REQUIREMENTS_FILENAME="requirements.txt"
REQUIREMENTS_STRLEN=17
REF_MODELS_ROOT="${REF_MODELS_ROOT:=../../../}"
SCRIPT_ROOT=".github/utils/onebom"
CWD=$(pwd)
REF_MODELS_ROOT_STRLEN=9
PYTHON_VERSION="3.10"
PREREQS_CONTAINER_NAME="amr-registry.caas.intel.com/aiops/airm-trivy-scan"
CSV_FILENAME="this.csv"
RM_VERSION="${RM_VERSION:=v3.0.0}"
ORIG_REQUIREMENTS_FILENAME="${REQUIREMENTS_FILENAME}.orig"
REQUIREMENTS_FILES_PATHS="requirements-txt-files.txt"
TRIVY_JSON_FILENAME="trivy-scan-spdx.json"
RUNNING_CONTAINER_NAME="sbom-scanner-container"
DEBUG=1

set -e

function debug () {
  if [[ "${DEBUG}" == "1" ]]
  then
    echo $1
  fi
}

function check_error () {
  if [[ "${1}" != "0" ]]
  then
    echo "########## ERROR: ${2} ################"
    exit ${1}
  fi
  echo "No Error"
}
