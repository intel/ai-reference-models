#!/bin/bash
# This script will generate a trivy scan for every requirements.txt file that it can find in the $REF_MODELS_ROOT

# Requirements:
# 1. docker must be installed
# 2. the script must be run as a user that can use docker
# 3. The script must be run as a user that can write files to the current directory

# Inputs:
# 1. The root of the AI Reference Models directory (Default: ../../../)
# 2. The Ref Models Version (Default: 'v3.0.0')
# 3. ./shared_vars.sh

# Outputs:
# 1. All requirements.txt files under $REF_MODELS_ROOT will be collated into a file called
#	requirements_collated.txt
# 2. All requirements will be frozen and written to a file in the current directory called
# 	requirements_collated.frozen.txt
# 3. All original requirements.txt files will be saved in their original locations as
# 	requirements.txt.orig
# 4. Each requirements.txt file will be replaced with a version that has all the requirements
#	locked to a version that is compatible with all of the other requirements in this
# 	version of the AI Reference Models. 
# 5. Trivy will scan each locked requirements.txt file and generate a file called 
#	<number>trivy-scan-spdx.json in the same directory as the requirements.txt file,
#	where <number> is a sequential identifier that prevents filename collisions when 
# 	uploading to https://goto.intel.com/scr-prod

source "./shared_vars.sh"

debug "HOST_PATH=${HOST_PATH}"

function generate-list-of-requirement-files () {
  IFS=$'\n' declare -g REQUIREMENTS_TXT_FILES=($(cd ${REF_MODELS_ROOT} && find . -name "${REQUIREMENTS_FILENAME}"))
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"

  # remove one if it exists
  rm -rf ${REQUIREMENTS_FILES_PATHS}

  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do 
    echo ${file} >> ${REQUIREMENTS_FILES_PATHS}
  done

  if [[ "${DEBUG}" == "1" ]]
  then
    debug "PWD: $(pwd)"
    debug "Paths to all requirements.txt files"
    cat ${REQUIREMENTS_FILES_PATHS}
  fi
}

generate-list-of-requirement-files