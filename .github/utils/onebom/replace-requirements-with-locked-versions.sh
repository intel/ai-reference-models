#!/bin/bash

# The output of this file is the set of original requirements .txt file and the set of 
# revised requirements.txt files that have versions locked. 

source "${HOST_PATH}/${SCRIPT_ROOT}/shared_vars.sh"

function generate_new_requirements_files () {
  #replace all versions with the frozen/reconciled version
  # Load the master requirements.txt file with the resolved and locked versions
  # metadata files are in script-root
  cd ${SCRIPT_ROOT}
  IFS=$'\n' declare requirements_versions=($(<${FROZEN_COLLATED_REQUIREMENTS_FILENAME}))
  if [[ "${DEBUG}" == "1" ]]
  then
    debug "Locked and Resolved Requirements versions:"
    echo ${requirements_versions[@]}
  fi

  # make an associative array to speed things up. 
  declare -A requirements_versions_map
  for vr in ${requirements_versions[*]}
  do
    local requirement_base=$(echo ${vr} | cut -d '=' -f 1)
    debug "Key: ${requirement_base}; Value: ${vr}"
    requirements_versions_map[${requirement_base}]="${vr}"
  done

# Get the list of all the paths to the original requirements.txt files
  IFS=$'\n' declare REQUIREMENTS_TXT_FILES=($(<${REQUIREMENTS_FILES_PATHS}))
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"
  debug "List of requirements.txt files: "
  printf "%s\n" "${REQUIREMENTS_TXT_FILES[@]}"

  # all paths are relative to REF_MODELS_ROOT
  cd ${REF_MODELS_ROOT}
  ls -la
# Replace all requirements with locked versions
  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do
    oIFS=$IFS
    IFS=$'\n'
    # Load the set of original requirements
    debug "PWD: $(pwd)"
    local original_requirements=($(<${file}))
    debug "Number of requirements in ${file}: ${#original_requirements[@]}"
    # backup the original file
    local backup_file="${file}.orig"
    cp "$file" "${backup_file}"
    rm -f "${file}"
    for r in ${original_requirements[*]}
    do
      local stripped_req=$(python3 ./${SCRIPT_ROOT}/split-multi-delim.py ${r})
      debug "Original Requirement: ${r}; Original without version: ${stripped_req}"
      if [[ "$stripped_req" != "" ]] 
      then
        local locked_req=requirements_versions_map[${stripped_req}]
        if [[ "${locked_req}" != "" ]]
        then
          debug "****match****"
          echo ${vr} >> ${file}
        else
          check_error -1 "No Value associated with Key: ${locked_req}"
        fi
      else
        debug "Stripped Requirement String was Blank."
      fi
      stripped_req=""
    done
    original_requirements=()
    #if we haven't written anything back to the requirements.txt file, something went wrong
    if [[ ! -f ${file} ]]
    then
      check_error -1 "${file} was not replaced with locked requirements.!"
    fi
  done
}

generate_new_requirements_files
