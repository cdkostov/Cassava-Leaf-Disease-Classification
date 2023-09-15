import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(600 * 800 * 3, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(torch.float32)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class ConvNet(nn.Module):
  def __init__(self):
    # Call nn.Module's constructor--don't forget this
    super().__init__()

    """
    Define layers
    """
    # Explanation of arguments
    # Remember a Convolution layer will take some input volume HxWxC
    # (H = height, W = width, and C = channels) and map it to some output
    # volume H'xW'xC'.
    #
    # Conv2d expects the following arguments
    #   - C, the number of channels in the input
    #   - C', the number of channels in the output
    #   - The filter size (called a kernel size in the documentation)
    #     Below, we specify 5, so our filters will be of size 5x5.
    #   - The amount of padding (default = 0)
    self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2) #notice how we use padding to prevent dimension reduction
    self.conv2 = nn.Conv2d(10, 15, kernel_size=3, padding=1)

    # Pooling layer takes two arguments
    #   - Filter size (in this case, 2x2)
    #   - Stride
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(15 * 75 * 100, 400)
    self.fc2 = nn.Linear(400, 100)
    self.fc3 = nn.Linear(100, 25)
    self.fc4 = nn.Linear(25, 5)

  def forward(self, x):
    # Comments below give the shape of x
    # n is batch size
    x = x.to(torch.float32)
    # (n, 3, 600, 800)
    #print("Conv1", x.shape)
    x = self.conv1(x)
    #print("After c1:", x.shape)
    x = F.relu(x)
    # (n, 10, 600, 800)
    #print("after relu:", x.shape)
    x = self.pool(x)
    # (n, 10, 300, 400)
    #print("Conv2", x.shape)
    x = self.conv2(x)
    #print("After c2:", x.shape)
    x = F.relu(x)
    # (n, 15, 300, 400)
    x = self.pool(x)
    x = self.pool(x)
    # (n, 15, 75, 100)
    x = torch.reshape(x, (-1, 15 * 75 * 100))
    # (n, 8 * 7 * 7)
    x = self.fc1(x)
    x = F.relu(x)
    # (n, 256)
    x = self.fc2(x)
    x = F.relu(x)
    # (n, 128)
    x = self.fc3(x)
    x = F.relu(x)
    # (n, 10)
    x = self.fc4(x)
    return x
