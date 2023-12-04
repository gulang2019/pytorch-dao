#include <DAO/generator.h>
#include <DAO/utils.h>
#include <DAO/globals.h>

namespace DAO {

DAO_API ConcurrentQueue<Kernel> kernel_queue = {};
DAO_API ConcurrentCounter kernel_counter = {};

void push_kernel(Kernel&& kernel)
{
  kernel._tid = gettid();
  // Create a lambda function that captures the original function and its arguments
  kernel_counter.increment();
  kernel_queue.push(std::move(kernel));
}

} // DAO 