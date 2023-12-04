## DAO 

### Overall system design 

![image](./system%20design.png)

### Task List 

- [ ] Run GPT-2 training; 
- [ ] Design memory allocator; 
- [ ] Develop memory allocator; 
- [ ] Develop memory prefetcher; 

### Profiling 

mnist torch 0.36 DAO 0.41 

### Code structure
- include: headers
- python: python plugin of DAO  
- test: c test of DAO
- testing: python test of DAO

### Build
Build a python environment 
```
cd ~/ssd/.venv
python -m venv ENV_NAME 
source ENV_NAME/bin/activate 
```

Build the project 
```
git clone git@github.com:gulang2019/pytorch-dao.git
cd pytorch-dao 
pip install -r requirements.txt 
USE_DAO=1 BUILD_CAFFE2=0 PRINT_CMAKE_DEBUG_INFO=1 CXX=/usr/bin/g++ CC=/usr/bin/gcc USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0  MAX_JOBS=30 DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 taskset --cpu-list 0-30 python setup.py develop 2>&1 | tee build.log
```

Build python frontend 
```
cd dao 
python -m pip install -e . 
```

Test
```
export PYTORCH_CUDA_ALLOC_CONF=backend:dao
cd dao/testing 
python ./add.py
```

Developing
```
# clean 
python setup.py clean
# incremental build 
USE_DAO=1 BUILD_CAFFE2=0 PRINT_CMAKE_DEBUG_INFO=1 CC=/usr/bin/gcc CXX=/usr/bin/g++ USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0  MAX_JOBS=30 DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 taskset --cpu-list 0-30 python setup.py develop 2>&1 | tee build.log
```

Testing 
```
python -c "import torch; c; b = a+a; print(b)" 
```

### Trouble shooting
- ImportError: /home/siyuanch/.conda/envs/dao/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30` not found when `import torch` 

This is because we use host gcc/g++ but conda libstdc++ ...

We can walk around this by modifying the `LD_LIBRARY_PATH` to instruct the compiler find the correct libstdc++.so
```
export LD_LIBRARY_PATH=
```
After that, your cmake may not find the `g++` or `gcc` when doing compilation. We can solve this by setting `CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake ..` with cmake.


- 'cc1plus' not found when building the torch
```
conda uninstall gcc
``` 

- ImportError: cannot import name 'Self' from 'typing_extensions' (/opt/anaconda3/lib/python3.8/site-packages/typing_extensions.py)
```
...pytorch-dao$ rm build/CMakeCache.txt
```

Test scripts for DAO 

```
cd dao 
mkdir build && cd build 
cmake ..
make
ctest --test-dir test
```
