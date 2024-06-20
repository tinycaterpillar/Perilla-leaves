import sys
import fourier_lib as lib
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18

# data config
N = 20000
dt = 0.000005
fs = N*10
ub = 25000

# hyper-param
nFFT = 128

# input
weights_path = input().strip()
val = list(map(float, sys.stdin.readline().strip().split(',')))
# with open("input_sample_ai.txt", 'r') as f:
#     weights_path = f.readline().strip()
#     val = list(map(float, f.readline().split(',')))
if len(val) != N: raise RuntimeError(f"input data size is not {N}")

obj = lib.Fourier_obj(val=val, dt=dt, fs=fs, nFFT=nFFT)

# AI
model = resnet18()
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())

# Load weights into the model
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

image = obj.get_verdict_data()
model.eval()
with torch.no_grad():
    outputs = model(image)
    output = torch.round(outputs) == 1
prob = outputs.item()
print(output.item(), f"{prob:.10f}")