#include <kernel-queue.h>
#include <stdarg.h>
namespace DAO {

void push_kernel(Kernel&& kernel) {
  // Create a lambda function that captures the original function and its arguments
  kernel_queue.push(std::move(kernel));
}

} // DAO 