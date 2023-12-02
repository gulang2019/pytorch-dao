#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <unordered_set>
#include <vector>


#include <DAO/globals.h>

namespace DAO {
// namespace c10 {
// namespace cuda {
// namespace CUDACachingAllocator {
// namespace CudaMallocAsync {

#if CUDA_VERSION >= 11040
// CUDA device allocator that uses cudaMallocAsync to implement
// the same interface as CUDACachingAllocator.cpp.

// Designed to be safe for CUDA graph capture.
// Interactions with CUDA graph capture are mediated by
// notifyCaptureBegin
// notifyCaptureAboutToEnd
// notifyCaptureEnded
// notifyCaptureDestroy

#define PINNED_MEMORY_SIZE (4 * 1024 * 1024 * 1024) // 4GB

Device::Device() {
  _used_bytes = 0; 
  size_t device_free, device_total; 
  cudaMemGetInfo(&device_free, &device_total);
  _memory_limits = device_free;
  _h2d_stream = c10::cuda::getStreamFromPool().stream();
  _d2h_stream = c10::cuda::getStreamFromPool().stream();
  _alignment = 128; 
  _pinned_size = PINNED_MEMORY_SIZE;
  C10_CUDA_CHECK(cudaHostAlloc(&_pinned_base_ptr, _pinned_size, cudaHostAllocDefault));
  DAO_ASSERT(_pinned_base_ptr != nullptr, "Failed to allocate pinned memory.");
  _base_ptr = nullptr;
  _base_size = 0;
  _initialized = false;
  DAO_INFO("Device pre initialize: id %d, memory limits %lu, pinned memory size %lu", 
      _device_index, _memory_limits, _pinned_size);
}

void Device::lazy_init(double fraction) {
  DAO_ASSERT(_initialized == false, "Device already initialized.");
  _initiailized = true;
  size_t free_size, total_size;
  C10_CUDA_CHECK(cudaMemGetInfo(&free_size, &total_size));
  _memory_limits = static_cast<uint64_t>(fraction * total_size);
  _base_size = std::min(_memory_limits, free_size);
  // reserve all free memory 
  C10_CUDA_CHECK(cudaMalloc(&_base_ptr, _base_size));
  DAO_INFO("Device post initialize: id %d, memory reserved %lu, pinned memory size %lu", 
      _device_index, _base_size, _pinned_size);
}

inline void sync_raw(cudaStream_t dependency, cudaStream_t dependent) {
  // CUDACachingAllocator.cpp uses raw cuda events, as do we.
  cudaEvent_t event = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event, dependency));
  C10_CUDA_CHECK(cudaStreamWaitEvent(dependent, event));
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

void dummy_deletor(void* ptr) {}
  
DataPtr DAOAllocator::allocate(size_t size) const override {
  return {nullptr, nullptr, &dummy_deletor, Device(DeviceType::CUDA, 0)};
}

DeleterFnPtr DAOAllocator::raw_deleter() const override {
  return &dummy_deletor;
}

void DAOAllocator::init(int dev_count) override {
  _device_count = dev_count; 
  _devices.resize(dev_count);
}

bool DAOAllocator::initialized() override {
  return _device_count > 0;
}

static inline void DAOAllocator::_assertValidDevice(int device) {
    DAO_ASSERT(
        0 <= device && device < _device_count, "Invalid device argument.");
}

void DAOAllocator::setMemoryFraction(double fraction, int device) override {
  TORCH_INTERNAL_ASSERT(
      0 <= fraction && fraction <= 1,
      "invalid fraction:",
      fraction,
      ". Please set within (0, 1).");

  std::lock_guard<std::mutex> lk(general_mutex);
  _assertValidDevice(device);
  c10::cuda::CUDAGuard g(device);
  _devices[device].lazy_init(fraction);
}

void DAOAllocator::emptyCache() override {
  DAO_ERROR("DAOAllocator::emptyCache() is not implemented but called.");
}

void DAOAllocator::cacheInfo(int device, size_t* maxWorkspaceGuess) override {
  *maxWorkspaceGuess = 0;
  DAO_ERROR("DAOAllocator::cacheInfo() is not implemented but called.");
}

void* getBaseAllocation(void* ptr, size_t* size) override {
  DAO_ERROR("DAOAllocator::getBaseAllocation() is not implemented but called.");
  return ptr;
}

void DAOAllocator::recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
  DAO_ERROR("DAOAllocator::recordStream() is not implemented but called.");
}

std::shared_ptr<void> DAOAllocator::getIpcDevPtr(std::string handle) override {
  DAO_ERROR("DAOAllocator::getIpcDevPtr() is not implemented but called.");
}

void DAOAllocator::recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) override {
  DAO_ERROR("DAOAllocator::recordHistory() is not implemented but called.");
}

void DAOAllocator::attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
  DAO_ERROR("DAOAllocator::attachOutOfMemoryObserver() is not implemented but called.");
}

void DAOAllocator::attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
  DAO_ERROR("DAOAllocator::attachAllocatorTraceTracker() is not implemented but called.");
}

std::shared_ptr<AllocatorState> DAOAllocator::getCheckpointState(int device, MempoolId_t id)
    override {
  DAO_ERROR("DAOAllocator::getCheckpointState() is not implemented but called.");
}

CheckpointDelta DAOAllocator::setCheckpointPoolState(
    int device,
    std::shared_ptr<AllocatorState> pps) override {
  DAO_ERROR("DAOAllocator::setCheckpointPoolState() is not implemented but called.");
}

// Collects stats for device.
// If device hasn't been used yet, returns 0s without creating a context.
DeviceStats DAOAllocator::getDeviceStats(int device) override {
  _assertValidDevice(device);

  // Memory currently reserved by the mempool
  uint64_t reserved_mem_current = 0;
  // High-water mark of memory reserved by the mempool since last reset
  uint64_t reserved_mem_peak = 0;
  // Memory currently in use by the mempool
  uint64_t used_mem_current = 0;
  // High-water mark of memory
  uint64_t used_mem_peak = 0;

  std::lock_guard<std::mutex> lk(general_mutex);

  if (_devices[device]._initialized) {
    c10::cuda::CUDAGuard g(device);

    cudaMemPool_t mempool = nullptr;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    C10_CUDA_CHECK(cudaMemPoolGetAttribute(
        mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem_current));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(
        mempool, cudaMemPoolAttrReservedMemHigh, &reserved_mem_peak));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(
        mempool, cudaMemPoolAttrUsedMemCurrent, &used_mem_current));

    C10_CUDA_CHECK(cudaMemPoolGetAttribute(
        mempool, cudaMemPoolAttrUsedMemHigh, &used_mem_peak));
  }

  // Many stat types are specific to the native allocator. We leave these
  // untouched. Their "struct Stat"s will contain zeroed values.
  DeviceStats stats;

  // In the native allocator:
  // allocated_bytes is the total bytes of blocks that have been malloc()ed
  // and not yet free()d.
  // active_bytes is the total bytes of blocks that have been malloc()ed but
  // not yet released back into a free pool. In other words, it includes all
  // allocated_bytes, as well as the bytes of "limbo state" blocks had have
  // already been free()ed but not yet free_block()ed back into a pool due to
  // outstanding stream_uses.
  //
  // Here, in the cudaMallocAsync allocator:
  // We simply ask the driver's opinion about active memory.
  // We don't bother distinguishing between allocated_bytes and active_bytes.
  stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
      used_mem_current;
  stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
      used_mem_peak;
  stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
      used_mem_current;
  stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
      used_mem_peak;
  stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
      reserved_mem_current;
  stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
      reserved_mem_peak;

  return stats;
}

void DAOAllocator::resetAccumulatedStats(int device) override {
  _assertValidDevice(device);
  TORCH_WARN_ONCE(
      "For backend:cudaMallocAsync, resetAccumulatedStats has no effect.");
}

void DAOAllocator::resetPeakStats(int device) override {
  _assertValidDevice(device);

  c10::cuda::CUDAGuard g(device);
  cudaMemPool_t mempool = nullptr;
  C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
  // Using zero as the reset value is the method recommended by Cuda driver
  // team. Vivek Kini says:
  //   "Resetting to zero (which is the only valid value when setting
  //    ReservedMemHigh) resets it to ReservedMemCurrent inside the driver
  //   (same goes for UsedMemHigh/UsedMemCurrent)"
  uint64_t zero = 0;
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(
      mempool, cudaMemPoolAttrReservedMemHigh, &zero));
  C10_CUDA_CHECK(
      cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrUsedMemHigh, &zero));
}

SnapshotInfo DAOAllocator::snapshot() override {
  TORCH_CHECK(
      false,
      "Calling snapshot with backend:cudaMallocAsync is not meaningful. "
      "(For backend:native, snapshot returns a detailed summary of all "
      "blocks tracked by the allocator, but the cudaMallocAsync backend "
      "does not track individual blocks.)");
  // Alternative: TORCH_WARN
  return {};
}

// CUDAGraph interactions
void DAOAllocator::beginAllocateStreamToPool(
    int device,
    cudaStream_t stream,
    MempoolId_t mempool_id) override {
  DAO_ERROR("DAOAllocator::beginAllocateStreamToPool() is not implemented but called.");
}

void DAOAllocator::endAllocateStreamToPool(int device, cudaStream_t) override {
  DAO_ERROR("DAOAllocator::endAllocateStreamToPool() is not implemented but called.");
}

void DAOAllocator::releasePool(int device, MempoolId_t mempool_id) override {
  DAO_ERROR("DAOAllocator::releasePool() is not implemented but called.");
}

void* DAOAllocator::raw_alloc(size_t nbytes) override {
  return nullptr; 
}

void* DAOAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
  return nullptr; 
}
void DAOAllocator::raw_delete(void* ptr) override {
  return nullptr; 
}
void DAOAllocator::enablePeerAccess(int dev, int dev_to_access) override {
  DAO_ERROR("DAOAllocator::enablePeerAccess() is not implemented but called.");
}

cudaError_t DAOAllocator::memcpyAsync(
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

void DAOAllocator::copy_data(void* dest, const void* src, std::size_t count) const final {
  C10_CUDA_CHECK(
      cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

std::string DAOAllocator::name() override {return "DAOAllocator";}

CudaMallocAsyncAllocator device_allocator;

void local_raw_delete(void* ptr) {
    return; 
}
CUDAAllocator* allocator() {
  return &device_allocator;
}

#else
CUDAAllocator* allocator() {
  TORCH_CHECK(false, "Cannot use cudaMallocAsyncAllocator with cuda < 11.4.");
  return nullptr;
}

#endif

// } // namespace CudaMallocAsync
// } // namespace CUDACachingAllocator
// } // namespace cuda
// } // namespace c10
} // namespace DAO 