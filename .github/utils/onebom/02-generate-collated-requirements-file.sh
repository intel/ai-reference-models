#!/bin/bash
# The output of this script is a file that has a concatenated list of the contents
# of all of the requirements.txt files.

source "./shared_vars.sh"

function collate_requirements () {
  # Load the list of paths to the requirements.txt files
  IFS=$'\n' declare REQUIREMENTS_TXT_FILES=($(<${REQUIREMENTS_FILES_PATHS}))
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"

  # create an assocative array to retain only unique elements
  rm -rf ${COLLATED_REQUIREMENTS_FILENAME}
  declare -A requirements_map

  #move up to the root directory
  cd ${REF_MODELS_ROOT}

  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do 
    debug "file = ${file}"
    local stripped="${file::-REQUIREMENTS_STRLEN}" 
    # collate the requirements
    oIFS=$IFS
    IFS=$'\n'
    local original_requirements=($(<${file}))
    debug "Number of requirements in ${file}: ${#original_requirements[@]}"
    for requirement in ${original_requirements[*]}
    do
      local stripped_req=$(python3 ./${SCRIPT_ROOT}/split-multi-delim.py ${requirement})
      # add the stripped requirement as a key to an associative arrary
      if [[ "$stripped_req" != "" ]]
      then
        requirements_map["${stripped_req}"]=1
      fi
    done
  done

  # write the unique set of requirements to the file in the CWD
  cd ${SCRIPT_ROOT}
  for requirement in ${!requirements_map[@]}
  do
    if [[ "$requirement" != "" ]]
    then
      debug "Unique requirement: ${requirement}"
      echo ${requirement} >> ${COLLATED_REQUIREMENTS_FILENAME}
    fi
  done
}

collate_requirements
