#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <DAO/executor.h>
#include <DAO/generator.h>
#include <DAO/globals.h>
#include <DAO/utils.h>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAFunctions.h>
namespace DAO {

extern ConcurrentQueue<Kernel> kernel_queue;
extern ConcurrentCounter kernel_counter;

namespace executor {

static std::thread executor_thread;

bool print_tensor(std::ostream& os, const at::Tensor& tensor) {  
  bool nan = false;
  bool all_zero = true;
  auto& storage_ptr = tensor.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr();
  os << "Tensor(" << tensor.toString() << "," << tensor.sizes() << "," << tensor.use_count() << ","; 
  std::cout << "data:"; 
  void* data = nullptr;
  if (storage_ptr.device().is_cpu()) {
    std::cout << "cpu, ";
    data = storage_ptr.get();
  } else if (storage_ptr.device().is_cuda()) {
    std::cout << "cuda, ";
    assert(storage_ptr.device().is_cuda());
    auto ptr = storage_ptr.get();
    size_t nbytes = tensor.nbytes();
    data = malloc(nbytes);
    assert(data);
    cudaMemcpy(data, ptr, nbytes, cudaMemcpyDeviceToHost);
  } else {
    DAO_ERROR("unknown device type");
  }
  if (tensor.scalar_type() == at::ScalarType::Float) {
    for (int i = 0; i < tensor.numel(); i++) {
      auto value = ((float*)data)[i];
      if (i < 5)
        os << value << ",";
      if (i > 100) break; 
      if (std::isnan(value)) {
        std::cout << "index " << i << " is " << value << ", "; 
        nan = true; 
        break; 
      }
      if (value != 0) all_zero = false;
    }
  } else if (tensor.scalar_type() == at::ScalarType::Int) {
    for (int i = 0; i < tensor.numel(); i++) {
      auto value = ((int*)data)[i];
      if (i < 5)
        os << value << ",";
      if (i > 100) break; 
      if (std::isnan(value)) {
        std::cout << "index " << i << " is " << value << ", "; 
        nan = true; 
        break; 
      }
      if (value != 0) all_zero = false;
    }
  }
  os << "storage:" << tensor.unsafeGetTensorImpl()->storage().use_count() << "," << storage_ptr.get() << "," << ")" << std::endl; 
  if (storage_ptr.device().is_cuda()) free(data);
  if (all_zero) DAO_WARNING("all zero");
  return nan || all_zero; 
}

void Executor::run() {
  while (true) {
    bool has_nan = false; 
    
    Kernel kernel = kernel_queue_.pop();
    DAO_INFO("Executor: %s, %d, %d in kernel_queue", kernel._name.c_str(), kernel._tid, kernel_queue.size());
    if (DAO::verbose > 1) {
      c10::cuda::device_synchronize();
      for (auto& tensor: kernel._inputs) {
        std::cout << "Input: ";
        bool nan = print_tensor(std::cout, tensor);
        if (nan) {
          DAO_WARNING("NaN or Inf detected");
          has_nan = true;
        }
      }
    }
    if (kernel.is_stop()) {
      DAO_INFO("Executor::run(): stop kernel");
      if (kernel_counter.peek()!=0) {
        DAO_WARNING("Executor: stop when kernel_counter is not zero");
        kernel_counter.set_zero();
      }
      break;
    }
    kernel._impl(&kernel); 
    kernel_counter.decrement();
    if (DAO::verbose > 1) {
      c10::cuda::device_synchronize();
      // for (auto& tensor: kernel._inputs) {
      //   std::cout << "Input: ";
      //   bool nan = print_tensor(std::cout, tensor);
      //   if (nan) {
      //     DAO_WARNING("NaN or Inf detected");
      //     has_nan = true;
      //   }
      // }
      for (auto& tensor: kernel._outputs) {
        std::cout << "Output: ";
        bool nan = print_tensor(std::cout, tensor);
        if (nan) {
          DAO_WARNING("NaN or Inf detected");
          has_nan = true;
        }
      }
    }
    if (has_nan) {
      DAO_ERROR("NaN or Inf detected");
    }
    // DAO_INFO("Executor::run(): decrement %s done", kernel._name.c_str());
  }
}

static bool launched = false;

void status() {
  DAO_INFO("status: launched %d kernel_counter(%p) = %d, %d in kernel_queue(%p)", 
    launched, &kernel_counter, kernel_counter.peek(), kernel_queue.size(), &kernel_queue);
}

void sync() {
  DAO_INFO("DAO::sync");
  status();
  if (launched) kernel_counter.wait_until_zero();
  cudaDeviceSynchronize();
}

void launch(){
  if (launched) {
    DAO_WARNING("executor has already been launched");
    return;
  }
  launched = true;
  DAO_INFO("launching kernel_queue address = %p", &kernel_queue);
  auto _entry = [](){
    Executor executor(kernel_queue); 
    executor.run();
  };
  executor_thread = std::thread(_entry);
  executor_thread.detach();
}

void stop() {
  DAO_INFO("DAO::stop");
  status();
  Kernel kernel;
  kernel.set_stop().set_name("stop");
  kernel_queue.push(std::move(kernel));
}

void log(const char* msg) {
  Kernel kernel;
  kernel.set_name(msg).set_impl([](Kernel* kernel){
    printf(ANSI_COLOR_BLUE "[DAO::Kernel Log]: %s\n" ANSI_COLOR_RESET, kernel->_name.c_str()); 
  });
  kernel_counter.increment();
  kernel_queue.push(std::move(kernel));
}

// void join(){
//   executor_thread.join();
// }

// void stop(){
//   // if (executor_thread.joinable())
//   DAO_ASSERT(executor_thread.joinable(), "executor thread is not joinable");
//   DAO_INFO("stopping executor thread");
//   executor_thread.std::thread::~thread();
// }

} // executor

} // DAO 