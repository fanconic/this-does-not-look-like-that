#!/bin/bash

clean () {
    trap SIGINT
    rm -rf /scratch/PPNet
    echo; echo 'Script terminated by user!!! Removed /scratch/PPNet.'
    exit                 
}

trap "clean" INT

echo 'Standard Training.'
python -u src/data/setup.py
python -u src/data/img_aug.py
python -u train.py

rm -rf /scratch/PPNet
echo 'Script completed! Removed /scratch/PPNet.'
