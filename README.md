# Installation
```shell
git clone git@github.com:Zouxxyy/cryoem-amino-acid-detection.git
cd cryoem-amino-acid-detection
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -e .


cd cuda_functions/nms_3D/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py

cd cuda_functions/roi_align_3D/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
```

# Prepare the Data
```shell
# dataset to use ['test', '400_500']
dataset='test'

python experiments/cryoEM_exp/preprocessing.py --EMdata_dir /mnt/data/zxy/amino-acid-detection/EMdata_dir/${dataset} \
                                               --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset}
```

# Train
```shell
# workspace
work_dir='xxxxxxx'
# dataset to use ['test', '400_500']
dataset='test'

if [[ ! -d "log" ]]; then
 mkdri log
fi

nohup python -u exec.py --mode train \
                        --exp_dir /mnt/data/zxy/amino-acid-detection/exec_dir/${work_dir} \
                        --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset} > ./log/${work_dir}.log 2>&1 &

tail -f log/${work_dir}.log
```

# Test
```shell
# workspace
work_dir='test'
# dataset to use ['test', '400_500']
dataset='test'
# weight_path for test
test_weight_path='/mnt/data/zxy/amino-acid-detection/exec_dir/1226/fold_0/71_best_checkpoint/params.pth'
# id for test
test_id_path='scrips/test_id.txt'

python exec.py --mode test \
               --exp_dir /mnt/data/zxy/amino-acid-detection/exec_dir/${work_dir} \
               --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset} \
               --test_weight_path ${test_weight_path} \
               --test_id_path ${test_id_path}
```
