#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# bash setup.sh MODE={stable,nightly,libtpu-only} LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu} DEVICE={tpu,gpu}

# You need to specificy a MODE, default value stable.
# You have the option to provide a LIBTPU_GCS_PATH that points to a libtpu.so provided to you by Google.
# In libtpu-only MODE, the LIBTPU_GCS_PATH is mandatory.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.13

# Save the script folder path of maxtext
# Enable "exit immediately if any command fails" option
 
# Code based on https://raw.githubusercontent.com/google/maxtext/main/setup.sh

set -e
export DEBIAN_FRONTEND=noninteractive

(sudo bash || bash) <<'EOF'
apt update  &&  \
apt install -y numactl &&  \
apt install -y lsb-release  &&  \
apt install -y gnupg  &&  \
apt install -y curl  &&  \
apt install -y aria2 && \
apt install -y jq && \
apt install -y git-lfs && \
apt install -y python3.10-venv
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# Write a requirements.txt so we don't have to upload it
FILE="requirements.txt"

/usr/bin/cat <<EOM >$FILE
# jax>=0.4.23
# jaxlib>=0.4.23
orbax-checkpoint>=0.5.2
absl-py
array-record
aqtp
cloud-tpu-diagnostics
google-cloud-storage
grain-nightly
flax>=0.8.0
ml-collections
numpy
optax
# protobuf>=4.25.3
pylint
pytest
pytype
sentencepiece==0.1.97
tensorflow-text>=2.13.0
tensorflow>=2.13.0
tensorflow-datasets
tensorboardx
tensorboard-plugin-profile
git+https://github.com/mlperf/logging.git
transformers
diffusers
datasets
jupyter
ipykernel
ipywidgets
ipython
matplotlib
wandb
huggingface_hub
yq
EOM

# Create a venv env and Source it
python -m venv venv
source venv/bin/activate

pip install -U pip

# Save current directory
run_name_folder_path=$(pwd)

# Uninstall existing jax, jaxlib, and libtpu-nightly
pip3 show jax && pip3 uninstall -y jax
pip3 show jaxlib && pip3 uninstall -y jaxlib
pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly

# Install nightly jax, jaxlib, and libtpu
echo "Installing jax-nightly, jaxlib-nightly, and libtpu-nightly for TPU"
pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip3 install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre

# Install nightly tensorboard plugin profile
echo "Installing nightly tensorboard plugin profile"
pip3 install tbp-nightly --upgrade

# Install dependencies from requirements.txt
pip3 install -U -r requirements.txt

# Install pre-commit hooks if available
[ -d ".git" ] && pre-commit install



# Install requirements.txt
cd $run_name_folder_path && pip install --upgrade pip &&  pip3 install -r requirements.txt

# Project specific installs

# Set up a development environment by cloning then running the following command in a virtual environment:
git clone https://github.com/baricev/diffusers.git && cd diffusers && git checkout -b instruct_pix2pix_flax && git pull origin instruct_pix2pix_flax && pip install -e ".[dev]"

cd $run_name_folder_path && git clone https://github.com/baricev/research_projects.git