### Mem Allocation 
This document illustrates the usage of the cuda allocator of pytorch. 

The code is listed in `c10/cuda/CUDACachingAllocator.cpp`. The API of the allocator exposed to the higher level logics are: 
- allocate; 
- free; 

The backend of the allocator is CudaMalloc and CUDAFree. The Caching Allocator allocate blocks of memory in two sizes (one small and one big) and would maintain blocks of the emptied memory.   

Internally, the CahingAllocator manages memory in blocks. There are 3 key logic: 
1. allocate(size); 
   1. Initialize the memory pools to empty; 
   2. The allocation policy finds the block with the minimum acceptable size .
   3. If found, the allocator split the memory if the block is oversized;  
   4. If such memory is not found, the allocator will try allocate new memory in two size. 
2. garbage collection; (OPTIONAL)
   1. Garbage Collection is trigured when the memory is not enough upon allocation; 
   2. The memory release policy is enabled by memorizing the gc_count, the number of times a block is not allocated before it is in the block pool, and release those memory with the oldest age.  
3. free; 
   1. Push the memory back to the pool;  


### Tensor and Operation;
When torch.Tensor is called; 
