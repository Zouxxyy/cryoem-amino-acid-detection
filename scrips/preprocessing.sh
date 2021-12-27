#!/bin/bash

dataset='400_500'

python experiments/cryoEM_exp/preprocessing.py --EMdata_dir /mnt/data/zxy/amino-acid-detection/EMdata_dir/${dataset} \
                                               --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset}
