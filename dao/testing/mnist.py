'''
you may need to pip install idx2numpy
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import idx2numpy
import os
from urllib.request import urlretrieve
import dao 

# dao.verbose(1)
dao.launch()

# Define a simple feed-forward model
class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Download the MNIST dataset
def download(filename):
    url = f"http://yann.lecun.com/exdb/mnist/{filename}"
    if not os.path.exists(filename):
        urlretrieve(url, filename)
    os.system('gzip -d ' + filename)

download('train-images-idx3-ubyte.gz')
download('train-labels-idx1-ubyte.gz')

# Load the MNIST dataset
images = idx2numpy.convert_from_file('train-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

# Normalize the images
images = images / 255.0

# Convert to PyTorch tensors
images = torch.tensor(images, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Create the model
model = FeedForward()
if torch.cuda.is_available():
    model = model.cuda()

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):  # 10 epochs
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    optimizer.zero_grad()
    output = model(images)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        dao.sync()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    
dao.stop()