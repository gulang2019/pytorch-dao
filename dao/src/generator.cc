#include <DAO/generator.h>
#include <DAO/utils.h>
#include <DAO/globals.h>

namespace DAO {

ConcurrentQueue<Kernel> kernel_queue;

void push_kernel(Kernel&& kernel) {
  if (DAO::verbose) {
    printf("generator: push_kernel %p\n", &kernel_queue);
  }   
  // Create a lambda function that captures the original function and its arguments
  kernel_queue.push(std::move(kernel));
}

} // DAO 