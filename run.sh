#!/bin/bash

python download.py \
    ../kinetics400/train.csv \
    ./output/ \
    --trim \
    --trim_dir trim_output \
    --videos-per-cat-path test_videos \
    --num-jobs 4
