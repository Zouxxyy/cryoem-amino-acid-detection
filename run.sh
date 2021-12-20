test_name='exe_dir_1220_0'
nohup python -u exec.py -m train --exp_dir /mnt/data1/zxy/TCIA/exec/${test_name} --exp_source experiments/cryoEM_exp > ./log/${test_name}.log 2>&1 &