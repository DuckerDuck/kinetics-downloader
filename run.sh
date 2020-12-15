#!/bin/bash

python download.py \
    ../kinetics400/train.csv \
    ./output/ \
    --trim \
    --videos-per-cat 10 \
    --num-jobs 1
