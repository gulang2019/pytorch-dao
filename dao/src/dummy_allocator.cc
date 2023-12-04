#include <DAO/globals.h>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

namespace DAO {

void dummy_deletor(void* ptr) {}

using namespace c10;
using namespace c10::cuda;
using namespace c10::cuda::CUDACachingAllocator;

struct DummyAllocator: public CUDAAllocator {
private:
      bool _initialized = false; 
      int _dev_count = 0;
      std::vector<double> _memory_fractions;
      mutable size_t total_allocated;
      mutable std::vector<void *> dummy_devPtrs;

public: 

      DataPtr allocate(size_t n) const override {
            DAO_INFO("allocate(%ld), allocated=%ld", n, total_allocated);
            int device = 0;
            C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
            void *p;
            cudaError_t e = cudaMalloc(&p, n);
            if (e) {
                  p = nullptr;
                  DAO_WARNING("cudaMalloc FAILED");
            } else {
                  total_allocated += n;
                  dummy_devPtrs.push_back(p);
            }
            return {p, p, &dummy_deletor, Device(DeviceType::CUDA, device)};
      }

      DeleterFnPtr raw_deleter() const override {
            return &dummy_deletor; 
      }

      
      void init(int dev_count) override {
            DAO_INFO("init(%d)", dev_count);
            _initialized = true;
            _dev_count = dev_count;_memory_fractions.resize(_dev_count);
            total_allocated = 0;
      }

      bool initialized() override {return _initialized;}

      void setMemoryFraction(double fraction, int device) override {_memory_fractions[device] = fraction;}

      void emptyCache() override {
            DAO_WARNING("emptyCache is not supported by DummyAllocator");
      }

      void cacheInfo(int device, size_t* maxWorkspaceGuess) override{
            DAO_WARNING("cacheInfo is not supported by DummyAllocator");
      }

      void* getBaseAllocation(void* ptr, size_t* size) override{
            DAO_WARNING("getBaseAllocation is not supported by DummyAllocator");
            if (size) *size = 0; 
            return nullptr;
      }

      void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
            DAO_WARNING("recordStream is not supported by DummyAllocator");
      }

      std::shared_ptr<void> getIpcDevPtr(std::string handle) override{
            DAO_WARNING("getIpcDevPtr is not supported by DummyAllocator");
            return nullptr;
      }

      void recordHistory(
            bool enabled,
            CreateContextFn context_recorder,
            size_t alloc_trace_max_entries,
            RecordContext when) override{
            DAO_WARNING("recordHistory is not supported by DummyAllocator");
      }

      void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override{
            DAO_WARNING("attachOutOfMemoryObserver is not supported by DummyAllocator");
      }

      void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
            DAO_WARNING("attachAllocatorTraceTracker is not supported by DummyAllocator");
      }
      std::shared_ptr<AllocatorState> getCheckpointState(int device, MempoolId_t id)
            override{
            DAO_WARNING("getCheckpointState is not supported by DummyAllocator");
            return nullptr;
      }

      CheckpointDelta setCheckpointPoolState(
            int device,
            std::shared_ptr<AllocatorState> pps) override{
            DAO_WARNING("setCheckpointPoolState is not supported by DummyAllocator");
            return CheckpointDelta();
      }

      DeviceStats getDeviceStats(int device) override{
            DAO_WARNING("getDeviceStats is not supported by DummyAllocator");
            return DeviceStats();
      }

      void resetAccumulatedStats(int device) override {
            DAO_WARNING("resetAccumulatedStats is not supported by DummyAllocator");
      }

      void resetPeakStats(int device) override{
            DAO_WARNING("resetPeakStats is not supported by DummyAllocator");
      }

      SnapshotInfo snapshot() override {
            DAO_WARNING("snapshot is not supported by DummyAllocator");
            return SnapshotInfo();
      }

      void beginAllocateStreamToPool(
            int device,
            cudaStream_t stream,
            MempoolId_t mempool_id) override{
            DAO_WARNING("beginAllocateStreamToPool is not supported by DummyAllocator");
      }

      void endAllocateStreamToPool(int device, cudaStream_t) override {
            DAO_WARNING("endAllocateStreamToPool is not supported by DummyAllocator");
      }

      void releasePool(int device, MempoolId_t mempool_id) override{
            DAO_WARNING("releasePool is not supported by DummyAllocator");
      }

      void* raw_alloc(size_t nbytes) override{
            DAO_WARNING("raw_alloc is not supported by DummyAllocator");
            return nullptr;
      }

      void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
            DAO_WARNING("raw_alloc_with_stream is not supported by DummyAllocator");
            return nullptr;
      }

      void raw_delete(void* ptr) override {
            DAO_WARNING("raw_delete is not supported by DummyAllocator");
      }

      void enablePeerAccess(int dev, int dev_to_access) override{
            DAO_WARNING("enablePeerAccess is not supported by DummyAllocator");
      }

      cudaError_t memcpyAsync(
            void* dst,
            int dstDevice,
            const void* src,
            int srcDevice,
            size_t count,
            cudaStream_t stream,
            bool p2p_enabled) override {
            if (p2p_enabled || dstDevice == srcDevice) {
            return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
            } else {
            return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
            }
      }

      std::string name() override {return "DummyAllocator";}

      void copy_data(void* dest, const void* src, std::size_t count) const final {
            C10_CUDA_CHECK(
            cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
      }
} dummy_allocator;

c10::cuda::CUDACachingAllocator::CUDAAllocator* getDummyAllocator() {
    return &dummy_allocator;
}

} // namespace DAO