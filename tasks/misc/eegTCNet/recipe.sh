#!/usr/bin/env bash
# Copyright 2023 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements_data.txt

if [ ! -d "eeg-tcnet-master" ]; then
  wget https://github.com/iis-eth-zurich/eeg-tcnet/archive/refs/heads/master.zip

  unzip master.zip
  rm master.zip
fi

if [ ! -d "data" ]; then
  mkdir data
  cd data || exit
  wget https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
  wget https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat

  cd ..
fi
python3 prepare_dataset.py --path=data

deactivate

rm -rf env

python3.7 -m venv env-quant
source ./env-quant/bin/activate
pip install --upgrade pip
pip install -r requirements_model.txt

python3.7 quantize_model.py

deactivate
rm -rf env-quant
