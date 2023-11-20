// concurrent-queue.h
#ifndef DAO_KERNEL_QUEUE_H_
#define DAO_KERNEL_QUEUE_H_

#include <vector> 
#include <functional> 

#include <ATen/Tensor.h> 
// #include <c10/util/Optional.h>

namespace DAO {

struct Kernel {
  std::function<void()> _impl;
  std::vector<at::Tensor> _inputs;
  std::vector<at::Tensor> _outputs;  

  Kernel& set_impl(std::function<void()> impl) {
    this->_impl = impl;
    return *this; 
  }

  template<typename... Args>
  Kernel& set_inputs(Args... args){
    (_inputs.push_back(args), ...);
    return (*this);
  }

  template<typename... Args>
  Kernel& set_outputs(Args...args) {
    (_outputs.push_back(args), ...);
    return (*this);
  }

  template<typename... Args>
  Kernel& set_optional_inputs(Args... args) {
    auto optional = [this](auto&& arg) {
      if (arg.has_value() && (*arg).defined()) {
        _inputs.push_back(arg.value());
      }
    };
    (optional(args), ...);
    return (*this);
  }
}; 

void push_kernel(Kernel&& kernel);
} // namespace DAO 
#endif // DAO_KERNEL_QUEUE_H_
