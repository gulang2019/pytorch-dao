#ifndef DAO_ALLOCATOR_H
#define DAO_ALLOCATOR_H

#include <unordered_map>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/Tensor.h> 

namespace DAO {

c10::cuda::CUDACachingAllocator::CUDAAllocator* getDummyAllocator();

typedef size_t logical_time_t;

struct TensorRecord {
    struct AccessPattern {
        bool isRead; 
        logical_time_t time;
    };
    bool evicted;     
    void* host_ptr;
    std::vector <AccessPattern> accesses; 
};

struct TensorTable {
public: 
    void update_access(at::Tensor& tensor); 
private: 
    unsigned _logical_time = 0;
    std::mutex _mutex;
    std::unordered_map<c10::TensorImpl*, TensorRecord> _table;
}; 

struct IntervalGraph {
};

struct Allocator {
    size_t _alignment; 
    void* _device_base;
    size_t _device_size;
    void* _host_base; 
    size_t _host_size;
    std::mutex _mutex; 
public: 
    void allocate(at::Tensor& tensor);
};

} // namespace DAO

#endif // DAO_ALLOCATOR_H
