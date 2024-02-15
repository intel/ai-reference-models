#!/bin/bash

# The working dir in the container is /host
source "./shared_vars.sh"

#install prereqs
apt-get update
apt-get install --no-install-recommends --fix-missing -y python${PYTHON_VERSION} pip git build-essential cmake wget apt-transport-https gnupg lsb-release
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -
echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | tee -a /etc/apt/sources.list.d/trivy.list
apt-get update && apt-get install trivy

pip install scikit-build
