Building the project 
```
conda create -n ENV_NAME python=3.9 
git clone git@github.com:gulang2019/pytorch-dao.git
cd pytorch-dao 
pip install -r requirements.txt 
USE_DAO=1 BUILD_CAFFE2=0 PRINT_CMAKE_DEBUG_INFO=1 CC=/usr/bin/gcc USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0  MAX_JOBS=30 DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 taskset --cpu-list 0-30 python setup.py develop 2>&1 | tee build.log
```

Developing
```
# clean 
python setup.py clean
# incremental build 
USE_DAO=1 BUILD_CAFFE2=0 PRINT_CMAKE_DEBUG_INFO=1 CC=/usr/bin/gcc USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0  MAX_JOBS=30 DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 taskset --cpu-list 0-30 python setup.py develop 2>&1 | tee build.log
```

Testing 
```
python -c "import torch; a = torch.Tensor([1,2,3], device='cuda'); b = a+a; print(b)" 
```

Trouble shooting
import torch error
```
conda install -c conda-forge gcc=12.1.0
``` 