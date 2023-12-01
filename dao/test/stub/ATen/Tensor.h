#ifndef ATEN_TENSOR_H_
#define ATEN_TENSOR_H_ 
#include <functional>
#include <string>
#include <vector>
#include <optional>
namespace at {

struct IntArrayRef {
    IntArrayRef() = delete;
    IntArrayRef(const std::vector<int64_t>& vec) : vec(vec) {}
    std::vector<int64_t> vec; 
};

typedef std::optional<IntArrayRef> OptionalIntArrayRef;

class Tensor{
public:
    bool defined() const {return true;}
    std::string toString() const {return "Tensor";}
    std::string name() const {return "Tensor";}
    int sizes() const {return 0;}
    int use_count() const {return 0;}
}; 
} // namespace at 

namespace c10{

typedef std::nullopt c10::nullopt; 

}

#endif // ATEN_TENSOR_H_