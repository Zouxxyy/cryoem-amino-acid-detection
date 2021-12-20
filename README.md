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
python experiments/cryoEM_exp/preprocessing.py
```

# Train
```shell
python exec.py -m train --exp_dir /mnt/data1/zxy/TCIA/exec/train_test --exp_source experiments/cryoEM_exp
```

# Test
```shell
python exec.py -m test --exp_dir /mnt/data1/zxy/TCIA/exec/predict_test --exp_source experiments/cryoEM_exp
```
