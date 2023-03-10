from torch import nn, cuda
import torch
import torch.nn.functional as F
import os
import pathlib

class Net(nn.Module):
    """MLP with 9 hidden(unit 8) layer with ReLU activate function"""
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, 8)
        self.l4 = nn.Linear(8, 8)
        self.l5 = nn.Linear(8, 8)
        self.l6 = nn.Linear(8, 8)
        self.l7 = nn.Linear(8, 8)
        self.l8 = nn.Linear(8, 8)
        self.l9 = nn.Linear(8, 8)
        self.l10 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = F.relu(self.l9(x))
        return self.l10(x)

device = "cuda" if cuda.is_available() else "cpu"
cur_path = str(pathlib.Path(__file__).parent.resolve())
model = Net()
model.to(device)

print(model.__doc__)
print(model)

model.load_state_dict(torch.load(cur_path + "/weight/3900_weight.pth", map_location=device))
print("load weight complete")