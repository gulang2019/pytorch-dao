import torch._C as dao 
import torch

dao.verbose(1)
dao.launch()
# assert torch.cuda.is_available()
# device = torch.device("cuda")

# Create two tensors 'a' and 'b'
a = torch.tensor([1, 2, 3], device='cuda')
b = torch.tensor([2,4,6], device='cuda')
# Perform element-wise addition
result = a + b

import time 

time.sleep(1)

dao.sync()
print(result)
# print(result)
