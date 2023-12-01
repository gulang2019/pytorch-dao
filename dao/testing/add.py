import dao

import torch

dao.launch()

assert torch.cuda.is_available()
device = torch.device("cuda")

# Create two tensors 'a' and 'b'
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)

# Perform element-wise addition
result = a + b

dao.sync()
print(result)

result_cpu = result.cpu()
print(result_cpu)