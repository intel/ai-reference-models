#!/bin/bash  
#
# Copyright (c) 2023 Intel Corporation
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

#

while true; do
terms_and_conditions=`cat terms_and_conditions.txt`
echo "$terms_and_conditions"

read -p "Do you agree to the terms and conditions? (y/n) " yn
echo 'unset -f USER_CONSENT' > .env
echo 'USER_CONSENT='$yn > .env

case $yn in
        [yY] ) echo Terms and conditions agreed;
        break;;

        [nN] ) echo Terms and conditions disagreed;
        exit;;

        * ) echo "Invalid Response. Please choose: y or n"
        exit 1;;

esac

done

# Install dependencies only if terms and conditions were agreed.
apt-get install -y wget
pip install -r requirements.txt
