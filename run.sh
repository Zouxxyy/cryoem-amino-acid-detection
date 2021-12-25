test_name='1224'
nohup python -u exec.py -m train --exp_dir /mnt/data/zxy/amino-acid-detection/exec_dir/${test_name} --exp_source experiments/cryoEM_exp > ./log/${test_name}.log 2>&1 &