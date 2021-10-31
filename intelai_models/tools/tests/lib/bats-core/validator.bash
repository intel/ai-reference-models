#!/usr/bin/env bash
#!/usr/bin/env bash
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


bats_test_count_validator() {
  header_pattern='[0-9]+\.\.[0-9]+'
  IFS= read -r header
  # repeat the header
  printf "%s\n" "$header"

  # if we detect a TAP plan
  if [[ "$header" =~ $header_pattern ]]; then
    # extract the number of tests ...
    local expected_number_of_tests="${header:3}"
    # ... count the actual number of [not ] oks...
    local actual_number_of_tests=0
    while IFS= read -r line; do
        # forward line
        printf "%s\n" "$line"
        case "$line" in
        'ok '*)
        (( ++actual_number_of_tests ))
        ;;
        'not ok'*)
        (( ++actual_number_of_tests ))
        ;;
        esac
    done
    # ... and error if they are not the same
    if [[ "${actual_number_of_tests}" != "${expected_number_of_tests}" ]]; then
        printf '# bats warning: Executed %s instead of expected %s tests\n' "$actual_number_of_tests" "$expected_number_of_tests"
        return 1
    fi
  else
    # forward output unchanged
    cat
  fi
}