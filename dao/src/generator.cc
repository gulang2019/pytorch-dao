#include <DAO/generator.h>
#include <DAO/utils.h>
#include <DAO/globals.h>

namespace DAO {

DAO_API ConcurrentQueue<Kernel> kernel_queue = {};
DAO_API ConcurrentCounter kernel_counter = {};

void push_kernel(Kernel&& kernel)
{
  DAO_INFO("Pushing kernel to queue");
  // Create a lambda function that captures the original function and its arguments
  kernel_queue.push(std::move(kernel));
  kernel_counter.increment();
}

} // DAO 