#!/bin/bash

FILENAME="app-needing-versions.txt"
declare -a items
mapfile -t items < "${FILENAME}"
old_IFS="$IFS"
IFS=$'\n'
for item in ${items[*]} ; do
  eval ${item} python3 onebom-create-speed-item.py
done
IFS="$old_IFS"