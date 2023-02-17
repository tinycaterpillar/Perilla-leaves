from torch import nn, cuda
import torch
import torch.nn.functional as F
import os
import pathlib

class Net(nn.Module):
    """MLP with 5 hidden layer with tanh activate function"""
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        x = torch.tanh(self.l5(x))
        return self.l6(x)

device = "cuda" if cuda.is_available() else "cpu"
cur_path = str(pathlib.Path(__file__).parent.resolve())
model = Net()

print(model.__doc__)
print(model)

model.load_state_dict(torch.load(cur_path + "/weight/20800_weight.pth", map_location=device))
print("load weight complete")