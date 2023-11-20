#include <thread>

#include <DAO/executor.h>
#include <DAO/generator.h>
#include <DAO/DAO.h>

namespace DAO {

extern ConcurrentQueue<Kernel> kernel_queue;

namespace executor {

static std::thread executor_thread;
static MutexBool is_running; 

void Executor::run() {
  while (true) {
    DAO_INFO("Executor::run(): popping kernel!");
    Kernel kernel = kernel_queue_.pop();
    DAO_INFO("Executor::run(): setting is_running");
    is_running.set();
    DAO_INFO("Executor::run(): run kernel");
    kernel._impl(); 
    DAO_INFO("Executor::run(): unsetting is_running");
    is_running.unset();
    DAO_INFO("Executor::run(): unsetting finished!");
  }
}

void sync() {
  DAO_INFO("sync");
  kernel_queue.wait_until_empty();
  is_running.wait_until(false);
}

void launch(){
  DAO_INFO("launching kernel_queue address = %p", &kernel_queue);
  auto _entry = [](){
    Executor executor(kernel_queue); 
    executor.run();
  };
  executor_thread = std::thread(_entry);
  executor_thread.detach();
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