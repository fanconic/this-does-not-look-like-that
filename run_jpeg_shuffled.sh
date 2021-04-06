#!/bin/bash

clean () {
    trap SIGINT
    rm -rf /scratch/PPNet
    echo; echo 'Script terminated by user!!! Removed /scratch/PPNet.'
    exit                 
}

trap "clean" INT

echo 'JPEG Training.'
python -u src/data/setup.py
python -u src/data/img_aug.py
python -u src/data/setup_compress.py
python -u train_jpeg_shuffled.py

rm -rf /scratch/PPNet
echo 'Script completed! Removed /scratch/PPNet.'
