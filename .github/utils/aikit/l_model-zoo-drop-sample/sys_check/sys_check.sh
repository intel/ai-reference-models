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


#each syscheck script should do five things:
#
# 1 - load the common.sh file   to gain the common API functions
# 2 - initialize an ERRORSTATE value to 0,
# 3 - return ERRORSTATE at the end of the shell script
# 4 - 'echo'   any problems out to the user. (and adjust ERRORSTATE if so)
# 5 - 'speak'  other messages to the user.

# 'echo' should be used for outputting messages in response to errors. 'echo' is always output.
# 'speak' outputs only if the -v verbose flag it used. Affirmative messages ( "Everything OK!" ) should
#    use 'speak', as well as advice, informative messages, or possibly longer explanations of an error. (eg "your cmake installation is not the latest" )
#
#  colors for use with 'echo' and 'speak' are defined.  See  common.sh for list and usage example.

# any arguments passed to the root syscheck script are passed on to this one.

# ERRORSTATE: 0 if OK, 1 if not.

#location of this sh file
LOC=$(realpath $(dirname "${BASH_SOURCE[0]}"))

#load common file
source $LOC/../../../common.sh "$@"

#every syscheck script should set up an ERRORSTATE variable and return it on completion.
ERRORSTATE=0

speak "Please use 'conda activate tensorflow' to activate Intel® Distribution for Tensorflow environment, if not already done."

#check if python is installed & also check for version
if [ ! -x "$(command -v python)" ]; then
    ERRORSTATE=1
    echo "Python is not available. Please installed python"
fi


return $ERRORSTATE
