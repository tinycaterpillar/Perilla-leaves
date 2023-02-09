from torch import nn, optim, from_numpy, cuda
import torch
import torch.nn.functional as F
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 8)
        self.l2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)

if __name__ == '__main__':
    directory = str(os.getcwd())
    model = Net()
    model.load_state_dict(torch.load(directory + "/weight.pth"))