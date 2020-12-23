#!/bin/bash

python download.py \
    ../kinetics400/train.csv \
    ./output/ \
    --trim \
    --videos-per-cat-path download_per_cat.txt \
    --num-jobs 1
