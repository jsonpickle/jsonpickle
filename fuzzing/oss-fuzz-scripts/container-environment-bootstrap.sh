#!/usr/bin/env bash

set -euo pipefail

#################
# Prerequisites #
#################

for cmd in python3 git wget rsync; do
  command -v "$cmd" >/dev/null 2>&1 || {
    printf '[%s] Required command %s not found, exiting.\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$cmd" >&2
    exit 1
  }
done

SEED_DATA_DIR="$SRC/seed_data"
mkdir -p "$SEED_DATA_DIR"

#############
# Functions #
#############

download_and_concatenate_common_dictionaries() {
  # Assign the first argument as the target file where all contents will be concatenated
  target_file="$1"

  # Shift the arguments so the first argument (target_file path) is removed
  # and only URLs are left for the loop below.
  shift

  for url in "$@"; do
    wget -qO- "$url" >>"$target_file"
    # Ensure there's a newline between each file's content
    echo >>"$target_file"
  done
}

fetch_seed_data() {
    rsync -avc "$SRC/jsonpickle/fuzzing/dictionaries/" "$SEED_DATA_DIR/"

    # Dogfood our own test files and use them as inputs data to seed the fuzzer!
    find "$SRC/jsonpickle/tests/" -type f -print | zip -jur "$SEED_DATA_DIR/__default_corpus.zip" -@
}

########################
# Main execution logic #
########################

fetch_seed_data

download_and_concatenate_common_dictionaries "$SEED_DATA_DIR/__base.dict" \
  "https://raw.githubusercontent.com/google/fuzzing/master/dictionaries/json.dict" \
  "https://raw.githubusercontent.com/google/fuzzing/master/dictionaries/jsonnet.dict" \
  "https://raw.githubusercontent.com/google/fuzzing/master/dictionaries/utf8.dict"

# The OSS-Fuzz base image has outdated dependencies by default so we upgrade them below.
python3 -m pip install --upgrade pip
# Upgrade to the latest versions known to work at the time the below changes were introduced:
python3 -m pip install 'setuptools~=69.0' 'pyinstaller~=6.0' typing_extensions
