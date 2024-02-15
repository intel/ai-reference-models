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

# The output of this script is the set of Trivy scans in JSON file for all of the requirements.txt
# files.

source "${HOST_PATH}/${SCRIPT_ROOT}/shared_vars.sh"

function do-trivy-scan () {
  debug "Scanning ${2} to ${1}"
  trivy fs --format spdx-json --list-all-pkgs --scanners vuln -o "${1}" "${2}"
  check_error $? "Error doing trivy scan ${2} to ${1}"
}

 # Append a counter to the end of the filename because the OneBoM SCR/Elements tool doesn't like it
# When we submit multiple trivy scans with the same filename.
function do-trivy-scans () {
  local j=0

  # Get the list of all the paths to the original requirements.txt files
  debug "PWD: $(pwd)"
  IFS=$'\n' declare REQUIREMENTS_TXT_FILES=($(<${SCRIPT_ROOT}/${REQUIREMENTS_FILES_PATHS}))
  debug "Number of requirements.txt files: ${#REQUIREMENTS_TXT_FILES[@]}"

  # clean up existing Trivy json files
  rm -rf "*${TRIVY_JSON_FILENAME}"

  for file in ${REQUIREMENTS_TXT_FILES[*]}
  do
    debug "file to be scanned: ${file}"
    if [[ -f "${file}" ]]
    then
      local stripped="${file::-REQUIREMENTS_STRLEN}"
      # do trivy scan
      local trivy_filename="${stripped}/${j}${TRIVY_JSON_FILENAME}"
      do-trivy-scan "${trivy_filename}" "${file}"
      set +e
      ((j++))
      set -e
    else
      check_error -1 "${file} does not exist at $(pwd)"
    fi
  done
}

do-trivy-scans