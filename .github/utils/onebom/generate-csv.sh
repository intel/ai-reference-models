#!/bin/bash

#set -xv

source "./shared_vars.sh"
CREATE_CSV_FILE=1

debug "HOST_PATH=${HOST_PATH}"

# Create CSV
CSV_FILENAME="${CSV_FILENAME:=SCR-input-values.csv}"
RM_VERSION="v3.0.0"
RELEASE_NAME="Intel(R) AI Reference Models "
RELEASE_VERSION=" ${RM_VERSION} Release"
SOFTWARE_DESCRIPTION="Scripts to run the AI workload found at "
NAMESPACE="frameworks_ai"
FIELDS="Primary Download Location,Software Name *,Software Description *,Namespace *,Triviy Scan SPDX Path,Other Download Locations,Version Internal,Version External,COO"
INNER_SOURCE_URI="https://github.com/intel-innersource/frameworks.ai.models.intel-models/blob/"
PUBLIC_URI="https://github.com/IntelAI/models/blob/"
COO="United States"

function write_header () {
  if [[ "${CREATE_CSV_FILE}" == "1" ]]
  then
    echo ${FIELDS} > ${CSV_FILENAME}
  fi
}

function write_fields () {
  #FIELDS="Primary Download Location,Software Name *,Software Description *,Namespace *,Triviy Scan SPDX Path,Other Download Locations,Version Internal,Version External,COO"
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"
  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do 
    #skip the root requirements.txt file
    if [[ ${file} == ${REF_MODELS_ROOT}${REQUIREMENTS_FILENAME} ]] 
    then 
      debug "Skipping root requirements.txt"
      continue
    fi
    debug "file: ${file}"
    local primary_download_location="${INNER_SOURCE_URI}${RM_VERSION}/${file:REF_MODELS_ROOT_STRLEN}"
    debug "primary_download_location: ${primary_download_location}"
    local app_name="${file:REF_MODELS_ROOT_STRLEN:(-REQUIREMENTS_STRLEN)}"
    local software_name="${RELEASE_NAME} - ${app_name}_src"
    debug "software_name: ${software_name}"
    local software_description="${SOFTWARE_DESCRIPTION} ${app_name}."
    debug "software_description: ${software_description}"
    local app_path="${file::(-REQUIREMENTS_STRLEN)}"
    debug "app_path: ${app_path}"
    local trivy_scan_file=$(find ${app_path} -name "*spdx.json" | head -n 1)
    if [ ! -e "${trivy_scan_file}" ]
    then
      error "trivy_scan_file does not exist for ${trivy_scan_file}"
      local trivy_scan_windows_path="NULL"
    else
      debug "trivy_scan_file: ${trivy_scan_file}"
      local trivy_scan_windows_path="$(echo ${trivy_scan_file:REF_MODELS_ROOT_STRLEN} | sed  's/\//\\/g')"
      debug "trivy_scan_windows_path: ${trivy_scan_windows_path}"
    fi

    local other_download_locations="${PUBLIC_URI}${RM_VERSION}/${file:REF_MODELS_ROOT_STRLEN}"
    debug "other_download_locations: ${other_download_locations}"
    local version_internal="${RM_VERSION}"
    local version_external="${RM_VERSION}"
    local coo="${COO}"
    echo "${primary_download_location},${software_name},${software_description},${NAMESPACE},${trivy_scan_windows_path},${other_download_locations},${RM_VERSION},${RM_VERSION},${COO}" >> "${CSV_FILENAME}"
  done
}



function create_files () {
  IFS=$'\n' declare -g REQUIREMENTS_TXT_FILES=($(find ${REF_MODELS_ROOT} -name "${REQUIREMENTS_FILENAME}"))
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"
  #header rows
  write_header
}

function clean_up () {
  # Now that trivy scans are done, restore the original requirements.txt files
  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do
    rm -f ${file}
    mv "${file}.orig" ${file}
  done
}


create_files
write_header
write_fields
if [[ "${DEBUG}" == "1" ]]
then 
  cat "${CSV_FILENAME}"
fi
#clean_up
echo "!!!!!! DONE !!!!!"
