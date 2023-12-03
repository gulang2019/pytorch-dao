#include <thread>
#include <stdio.h>

#include <DAO/executor.h>
#include <DAO/generator.h>
#include <DAO/globals.h>
#include <DAO/utils.h>

#include <cuda_runtime.h>

namespace DAO {

extern ConcurrentQueue<Kernel> kernel_queue;
extern ConcurrentCounter kernel_counter;

namespace executor {

static std::thread executor_thread;

void Executor::run() {
  while (true) {
    Kernel kernel = kernel_queue_.pop();
    DAO_INFO("Executor: %s, %d in kernel_queue", kernel._name.c_str(), kernel_queue.size());
    if (kernel.is_stop()) {
      DAO_INFO("Executor::run(): stop kernel");
      break;
    }
    kernel._impl(&kernel); 
    kernel_counter.decrement();
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
  kernel.set_stop();
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