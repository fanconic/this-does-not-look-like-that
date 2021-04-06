#!/bin/bash

clean () {
    trap SIGINT
    rm -rf /scratch/PPNet
    echo; echo 'Script terminated by user!!! Removed /scratch/PPNet.'
    exit                 
}

trap "clean" INT

echo 'Fast FGSM Training.'
python -u src/data/setup.py
python -u src/data/img_aug.py
python -u train_fgsm.py

rm -rf /scratch/PPNet
echo 'Script completed! Removed /scratch/PPNet.'
