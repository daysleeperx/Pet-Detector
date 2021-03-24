#!/bin/bash

sudo pip3 install virtualenv
python3 -m venv iti0215-pet-detector
source iti0215-pet-detector/bin/activate
bash get_pi_requirements.sh
python3 adapter.py
