import torch
import torch.nn as nn
import dao 

torch.random.manual_seed(42)
# Define a simple feed-forward model
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        dao.status()
        dao.sync()
        dao.status()
        x = self.output(x)
        return x


dao.verbose(1)
dao.launch()
# Create the model
model = FeedForward(input_dim=10, hidden_dim=50, output_dim=1).cuda()
# import random 

# Create a random tensor to represent input data
input_data = torch.randn(32, 10).cuda()  # batch size = 32, input dimension = 10

# Create a random tensor to represent target data
target_data = torch.randn(32, 1).cuda()  # batch size = 32, output dimension = 1

# Forward pass through the model
output_data = model(input_data)

# Compute the MSE loss
# loss_fn = nn.MSELoss().cuda()
# loss = loss_fn(output_data, target_data)

loss = (output_data - target_data).pow(2).mean()

print('requires_grad', loss.requires_grad)

dao.sync()

print(f"Loss: {loss.item()}")

dao.stop()