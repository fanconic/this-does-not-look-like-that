#!/bin/bash

python -u setup.py
python -u img_aug.py
python -u main.py

rm -rf /scratch/PPNet

