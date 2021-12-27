#!/bin/bash

work_dir='test'
dataset='test'
test_weight_path='/mnt/data/zxy/amino-acid-detection/exec_dir/1226/fold_0/71_best_checkpoint/params.pth'
test_id_path='scrips/test_id.txt'

python exec.py --mode test \
               --exp_dir /mnt/data/zxy/amino-acid-detection/exec_dir/${work_dir} \
               --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset} \
               --test_weight_path ${test_weight_path} \
               --test_id_path ${test_id_path}
