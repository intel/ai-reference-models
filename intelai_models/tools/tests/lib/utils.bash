#!/bin/bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



# The regex for the "try" key word
try_regex_verify="^at +most +([0-9]+) +times +every +([0-9]+)s +to +get +([a-z]+) +named +'([^']+)' +and +verify +that +'([^']+)' +is +'([^']+)'$"
try_regex_find="^at +most +([0-9]+) +times +every +([0-9]+)s +to +find +([0-9]+) +([a-z]+) +named +'([^']+)' +with +'([^']+)' +being +'([^']+)'$"

# The regex for the "verify" key word
verify_regex_count_is="^there +is +(0|1) +([a-z]+) +named +'([^']+)'$"
verify_regex_count_are="^there +are +([0-9]+) +([a-z]+) +named +'([^']+)'$"
verify_regex_property_is="^'([^']+)' +is +'([^']+)' +for +([a-z]+) +named +'([^']+)'$"



# Prints a string in lower case.
# @param {string} The string.
# @return 0
to_lower_case() {
	echo "$1" | tr '[:upper:]' '[:lower:]'
}


# Trims a text.
# @param {string} The string.
# @return 0
trim() {
	echo $1 | sed -e 's/^[[:space:]]*([^[[:space:]]].*[^[[:space:]]])[[:space:]]*$/$1/'
}


# Trims ANSI codes (used to format strings in consoles).
# @param {string} The string.
# @return 0
trim_ansi_codes() {
	echo $1 | sed -e 's/[[:cntrl:]]\[[0-9;]*[a-zA-Z]//g'
}


# Adds a debug message for a given test.
# @param {string} The debug message.
# @return 0
debug() {
	debug_filename=$(basename -- $BATS_TEST_FILENAME)
	mkdir -p /tmp/detik
	echo -e "$1" >> "/tmp/detik/$debug_filename.debug"
}


# Deletes the file that contains debug messages for a given test.
# @return 0
reset_debug() {
	debug_filename=$(basename -- $BATS_TEST_FILENAME)
	rm -f "/tmp/detik/$debug_filename.debug"
}


# Adds a debug message for a given test about DETIK.
# @param {string} The debug message.
# @return 0
detik_debug() {

	if [[ "$DEBUG_DETIK" == "true" ]]; then
		debug "$1"
        fi
}


# Deletes the file that contains debug messages for a given test about DETIK.
# @return 0
reset_detik_debug() {

	if [[ "$DEBUG_DETIK" == "true" ]]; then
		reset_debug
	fi
}
