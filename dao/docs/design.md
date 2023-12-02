We need to maintain a table to record each tensors' lifetime in logical time axis.


## Algorithm; 

We have two threads; one for generator, the other for executor, allocator; 
Shared object between two threads: 
- tensor access patterns: a map from tensor to its access sequence <logical time, size, Read/Write> 
- kernel queue: a queue of kernel; 



Object types:
```python

[[VISIBILITY: thread 1,2]]
class TensorAccess:
    logical_time_t t;
    access_type_t type in {READ, WRITE};
[[VISIBILITY: thread 1,2]]
class TensorAccesses: 
    map<TensorImpl*, std::vector<TensorAccess>> access_patterns_;
    
    THREAD_SAFE def updateAccess(tensor: TensorImpl*, acc: access_type_t , t: logical_time_t):
        self.access_patterns_[tensor].push_back(TensorAccess(acc, t))

[[VISIBILITY: thread 1,2]]
class Kernel:
    std::vector<TensorRecord> inputs;
    std::vector<TensorRecord> outputs;
    void() impl;
    def set_inputs(std::vector<at::Tensor> tensors) {
        for tensor in tensors: 
            TensorRecord record; 
            record.use_count = tensor.use_count();
            record.ptr = tensor.unsafeGetImpl();
            record.size = tensor.size();
            inputs.push_back(tensor);
    }
    
    def set_outputs(std::vector<at::Tesnor> tensors) {
        for tensor in tensors: 
            TensorRecord record; 
            record.use_count = tensor.use_count();
            record.ptr = tensor.unsafeGetImpl();
            record.size = tensor.size();
            outputs.push(tensor);
    }

[[VISIBILITY: thread 1,2]]
class TensorRecord: 
    TensorImpl* ptr; # an unique identifier;
    size_t size; 
    size_t use_count; 
    Block* device_block = null;
    Block* host_block = null;
    status_t status in {ONCPU, ONDEVICE, UNINITIALIZED} = UNINITIALIZED; 

[[VISIBILITY: thread 2]]
class TensorTable: 
    map<TensorImpl*, TensorRecord> tensors; 
    def find(self, tensor: TensorRecord) -> TensorRecord:
        if (tensor = tensors.find(tensor.ptr)) == null:
            tensor = tensors.insert(record.ptr, record)
        return tensor
    def free(self, tensor: TensorRecord):
        tensors.erase(tensor->ptr)

class Block: 
    TensorRecord* record; 
    bool free; 
    void* start_ptr;
    size_t size; 

```

Thread1: Generator

```python 
def A-Torch-Func(inputs: std::vector<at::Tensor>, outputs: std::vector<at::Tensor>):
    op = new op; 
    op->meta(inputs, outputs);
    DAO::Kernel kernel; 
    kernel.set_inputs(inputs);
    kernel.set_outputs(outputs);
    kernel.set_impl([](){op->impl(inputs, outputs)});
    DAO::push_kernel(kernel);
    return 

def DAO::push_kernel(kernel: DAO::Kernel):    
    static logical_time_t t = 0;
    t += 1; 
    for input in kernel.inputs:
        tesnor_accesses.updateAccess(input, R, t); 
    for output in kernel.outputs:
        tesnor_accesses.updateAccess(output, W, t); 
    kernel_queue.push(Kernel(inputs, output, func));
```

Thread2: 
```python
# always on executor; 
class Executor: 
    def __init__(self):
        self.allocator = Allocator()
    # this is always on by dao.launch()
    def Execute(self): 
        While (True): 
            kernel = kernel_queue.pop()
            for input: TensorRecord in kernel.inputs:
                self.allocator.prepare(input) 
            for output: TensorRecord in kernel.outputs:
                self.allocator.prepare(output)
            kernel.execute();
            for input in kernel.inputs:
                if (LastAccess(input)) 
                    self.Allocator.free(tensor) 

# Allocator A member of the executor; 
class Allocator: 
    def __init__(self, initReservedDeviceMemory, initReservedHostMemory):
        self.host_mem = CPUMemoryManager(initReservedHostMemory)
        self.device_mem = GPUMemoryManager(initDeviceMemory, self.host_mem)
        self.tensor_table = new TensorTable() 
    def prepare(self, tensor_record: TensorRecord): 
        record = self.tensor_table.find(tensor_record) 
        if record.status is ONDEVICE: return 
        suc = self.device_mem.allocate(tensor_record) 
        
        if !suc: raise OOMError 

        if record.status is ONCPU:
            self.device_mem.fetch(tensor_record)
    
    def free(self, tensor_record):
        self.tensor_table.free(tensor_record) 
        self.device_mem.free_block(tensor_record.device_block)
        self.host_mem.free_block(tensor_record.host_block)

```python
class MemoryManager:
    def __init__(self, reserved):
        # link list of memory chunk sorted by memory size 
        self.blocks = Block(self.malloc(reserved)); 
    
    # A allocate function works for both cpu and gpu 
    def allocateOnDevice(self, tensor: tensorRecord, block_ptr: Block**) -> Bool: 
        for block in self.blocks:   
            if block.free and block.size > tensor.size: 
                block.free = False; 
                block.record = &tensor; 
                *block_ptr = block 
                return True 
        block = new Block()
        self.malloc(&block.start, tensor.size)
        if block == NULL: return False 
        block.size = tensor.size 
        block.record = &tensor 
        block.free = False
        *block_ptr = block; 
        # insert in to the blocks, keep blocks sorted; 
        self.blocks.insert(block)
        return True 
    
    def free_block(self, block* block):
        CheckBlockExists(self.block, self.blocks) 
        self.block.free = True 
        [Optional] Merge(self.blocks, self.block)

    virtual def  allocate(self, tensor: TensorRecord) -> Bool = 0; 
    virtual def malloc(self, void**ptr, size) = 0; 
    virtual def free(self, ptr) = 0;

class GPUMemoryManager: public MemoryManager 
    def __init__(self, backupMemory: CPUMemoryManager, stream):
        self.backupMemory = backupMemory 
        self.evictScoreFunction = EvictScoreFunc(tensor_access_patterns); 
        self.deviceToHostStream = createStream()
        self.HostToDeviceStream = createStream()
        self.computeStream = getComputeStream()
    
    def allocate (self, tensor: TensorRecord): override
        suc = self.allocateOnDevice(tensor, &tensor->device_ptr) 
        if suc: 
            tensor->status = ONDEVICE 
            return True; 
    
        evictCandidates = [self.blocks[i:j] s.t. sum(blocks[i:j].size()) > tensor.size and blocks[i:j] is consecutive] 
        evictBlocks = argmin_{blocks in evictCandidates}(evictScoreFunction(blocks)) 
        for block in evictBlocks: 
            suc = self.backupMemory.allocate(block.record) 
            raise HostOutOfMemoryError
            if !block.free: 
                block->record.status = ONCPU; 
                MemcpyAsync(Device2Host, device2HostStream, tensor.device_block, tensor.host_block)
                sync(device2HostStream, ComputeStream)
        
        block = self.blocks.merge(evictBlocks)
        tensor.block = block
        block.record = &tensor
        block.free = False 
        tensor.status = ONDEVICE;
        return True 

    def fetch(self, tensor: TensorRecord):
        tensor.status = ONDEVICE 
        MemcpyAsync(HostToDevice, self.HostToDeviceStream, tensor.host_block, tensor.device_block)
        sync(HostToDeviceStream, computeStream) 
        self.backupMemory.free(tensor.host_block) 

    def _malloc(self, size): 
        return CUDAMalloc(size) 

    def _free(self, ptr):
        CUDAfree(ptr);

class CPUMemoryManager: public MemoryManager
    def allocate(self, tensor: TensorRecord) -> Bool = 0; 
        suc = self.allocateOnDevice(tensor, &tensor->host_ptr)  
        if suc: return; 
        

class evictScoreFunction:
    def __init__(self, tensorAccess: TensorAccesses):
        self.tensorAccess = tensorAccess 
    def __call__(self, blocks):
        score: float = computeSocre(self, blocks)
        return float(score)
    # A score function to map the evicted blocks to a score according to their access pattern;  
    virtual computeScore(self, blocks) = 0; 
``` 