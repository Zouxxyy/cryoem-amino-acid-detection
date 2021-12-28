#!/bin/bash

work_dir='1228'
dataset='400_500'

if [[ ! -d "log" ]]; then
 mkdri log
fi

nohup python -u exec.py --mode train \
                        --exp_dir /mnt/data/zxy/amino-acid-detection/exec_dir/${work_dir} \
                        --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset} > ./log/${work_dir}.log 2>&1 &

tail -f log/${work_dir}.log
