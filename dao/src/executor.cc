#include <executor.h>

namespace DAO {

void Executor::run() {
  while (true) {
    Kernel kernel = kernel_queue_.pop(); 
    is_running.set();
    kernel._impl(); 
    is_running.unset();
  }
}

void executor_entry() {
  Executor executor(kernel_queue); 
  executor.run();
}

void synchronize() {
  kernel_queue.wait_until_empty();
  is_running.wait_until(false);
}

} // DAO 