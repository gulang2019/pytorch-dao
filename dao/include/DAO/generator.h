// concurrent-queue.h
#ifndef DAO_KERNEL_QUEUE_H_
#define DAO_KERNEL_QUEUE_H_

#include <vector> 
#include <functional> 
#include <optional>
#include <string> 

#include <ATen/Tensor.h> 
#include <ATen/Scalar.h> 
// #include <c10/util/Optional.h>

#include <DAO/globals.h>

namespace DAO {

struct Kernel {
  std::function<void(Kernel*)> _impl;
  std::vector<at::Tensor> _inputs;
  std::vector<at::Tensor> _outputs;  
  std::vector<at::Scalar> _scalars;
  std::vector<std::vector<int64_t> > _arrays;
  std::string _name = ""; 
  bool _stop = false; 

  Kernel& set_impl(std::function<void(Kernel*)> impl) {
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

  template<typename... Args> 
  Kernel& set_scalars(Args...args) {
    (_scalars.push_back(args), ...);
    return *this;
  } 

  template<typename... Args> 
  Kernel& set_optional_scalars(Args...args) {
    auto optional = [this](auto&& arg) {
      if (arg.has_value()) {
        _scalars.push_back(arg.value());
      }
    };
    (optional(args), ...);
    return *this;
  }

  int set_optional_array(std::optional<std::vector<int64_t>> array) {
    if (array.has_value()){
      _arrays.push_back(array.value());
      return _arrays.size()-1;
    } 
    return -1;
  }

  int set_optional_array(at::OptionalIntArrayRef ref) {
    if (ref.has_value()) {
      _arrays.push_back(ref.value().vec());
      return _arrays.size()-1;
    }
    return -1; 
  }

  at::OptionalIntArrayRef get_optional_array(int idx) {
    if (idx == -1) return c10::nullopt;
    assert(idx >= 0 && idx < (int)_arrays.size());
    return _arrays[idx];
  }

  int set_array(at::IntArrayRef ref) {
    _arrays.push_back(ref.vec());
    return _arrays.size()-1;
  }

  at::IntArrayRef get_array(int idx) {
    assert(idx >= 0 && idx < (int)_arrays.size());
    return _arrays[idx];
  }

  Kernel& set_stop() {
    _stop = true;
    return *this;
  }

  bool is_stop() const {
    return _stop;
  }

  Kernel& set_name(const char* name) {
    _name = std::string(name);
    return *this;
  }
}; 

DAO_API void push_kernel(Kernel&& kernel);

} // namespace DAO 
#endif // DAO_KERNEL_QUEUE_H_
