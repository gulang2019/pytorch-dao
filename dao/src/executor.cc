#include <thread>
#include <stdio.h>

#include <DAO/executor.h>
#include <DAO/generator.h>
#include <DAO/globals.h>
#include <DAO/utils.h>


namespace DAO {

extern ConcurrentQueue<Kernel> kernel_queue;
extern ConcurrentCounter kernel_counter;

namespace executor {

static std::thread executor_thread;

void Executor::run() {
  DAO_INFO("Executor: kernel_counter(%p) = %d, %d in kernel_queue(%p)", 
    &kernel_counter, kernel_counter.peek(), kernel_queue.size(), &kernel_queue);
  while (true) {
    DAO_INFO("Executor::run(): popping kernel!");
    Kernel kernel = kernel_queue_.pop();
    if (kernel.is_stop()) {
      DAO_INFO("Executor::run(): stop kernel");
      break;
    }
    std::cout << "input tensors: ";
    for (auto& tensor : kernel._inputs) {
      std::cout << tensor.toString() << tensor.sizes() << tensor.use_count() << ", ";
    }
    std::cout << std::endl;
    // std::cout << "input scalars: ";
    // for (auto& scalar : kernel._scalars) {
    //   std::cout << scalar.toString() << ", ";
    // }
    std::cout << std::endl;

    DAO_INFO("Executor::run(): run kernel");
    kernel._impl(); 
    DAO_INFO("Executor::run(): update kernel counter");
    kernel_counter.decrement();
  }
}

void status() {
  DAO_INFO("status: kernel_counter(%p) = %d, %d in kernel_queue(%p)", 
    &kernel_counter, kernel_counter.peek(), kernel_queue.size(), &kernel_queue);
}

void sync() {
  kernel_counter.wait_until_zero();
}

void launch(){
  static bool launched = false;
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