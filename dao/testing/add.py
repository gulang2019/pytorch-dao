import dao

import torch

# Create two tensors 'a' and 'b'
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Perform element-wise addition
result = a + b

print(result)