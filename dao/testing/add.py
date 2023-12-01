import dao
import torch

dao.verbose(1)
dao.launch()
# assert torch.cuda.is_available()
# device = torch.device("cuda")

# Create two tensors 'a' and 'b'
a = torch.tensor([1.0,2.0,3.0], device='cuda')
b = torch.tensor([2.0,4.0,6.0], device='cuda')
# Perform element-wise addition
result = a + b

dao.sync()
dao.stop()
# print('result', result)
# print(result)
