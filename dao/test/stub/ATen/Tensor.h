#ifndef ATEN_TENSOR_H_
#define ATEN_TENSOR_H_ 

#include <string>

namespace at {
class Tensor{
public:
    bool defined() const {return true;}
    std::string toString() const {return "Tensor";}
    std::string name() const {return "Tensor";}
    int sizes() const {return 0;}
    int use_count() const {return 0;}
}; 
} // namespace at 

#endif // ATEN_TENSOR_H_