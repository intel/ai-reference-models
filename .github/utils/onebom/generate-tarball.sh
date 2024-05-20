#!/bin/bash

#this script is not run in the container

source "./shared_vars.sh"

TARBALL_FILENAME="${TARBALL_FILENAME:=SCR-trivy-scans.tgz}"
FILE_LIST_TXT="${FILE_LIST_TXT:=trivy-file-list.txt}"
FILE_PATTERN="${FILE_PATTERN:=*spdx.json}"

function create_file_list () {
  # metadata is in script-root
  debug "FILE_PATTERN: ${FILE_PATTERN}"
  debug "TARBALL_FILENAME: ${TARBALL_FILENAME}"
  debug "FILE_LIST_TXT: ${FILE_LIST_TXT}"
  IFS=$'\n' declare -g FOUND_FILES=($(find . -name "${FILE_PATTERN}"))
  debug "Number of files to include in tarball: ${#FOUND_FILES[@]}"
  debug "Files to include: ${FOUND_FILES[@]}"
  rm -f "${FILE_LIST_TXT}"
  for file in "${FOUND_FILES[@]}"
  do
    echo "${file}" >> "${FILE_LIST_TXT}"
  done
  #header rows
}

function create_tarball () {
  tar -czvf "${TARBALL_FILENAME}" -T "${FILE_LIST_TXT}"
}

cd ${REF_MODELS_ROOT}
 debug "PWD: $(pwd)"
create_file_list
if [[ "${DEBUG}" == "1" ]]
then 
  echo "${FILE_LIST_TXT}"
  cat "${FILE_LIST_TXT}"
fi
create_tarball
check_error $? "Error creating tarball $TARBALL_FILENAME"
ls -l
debug "PWD: $(pwd)"
