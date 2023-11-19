#include <executor.h>

namespace DAO {

void Executor::run() {
  while (true) {
    Kernel kernel = kernel_queue_.pop(); 
    kernel._impl(); 
  }
}

void executor_entry() {
  Executor executor(kernel_queue); 
  executor.run();
}

} // DAO 