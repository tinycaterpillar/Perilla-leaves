import sys
import torch
import fourier_lib as lib

# data config
N = 20000
dt = 0.000005
fs = N*10
ub = 25000

# hyper-param
nFFT = 128

# input
# weights_path = input().strip()
# data_path = input().strip()
with open("input_sample_ai.txt", 'r') as f:
    weights_path = f.readline().strip()
    data_path = f.readline().strip()

df = lib.load_data(data_path)
ai_input = []
lengths = [len(df.columns)]
for c in df.columns:
    obj = lib.Fourier_obj(val=df[c][:N], dt=dt, fs=fs, nFFT=nFFT)
    ai_input.append(obj.get_verdict_data())

while len(ai_input) < 18:
    ai_input.append(torch.zeros((3, 300, 300)))
ai_input = torch.stack(ai_input).unsqueeze(0)
lengths = torch.tensor(lengths)

# AI
model = lib.get_model()

# Load weights into the model
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

model.eval()
with torch.no_grad():
    outputs = model(ai_input, lengths)
    outputs = 1 - outputs
    output = torch.round(outputs) == 1
prob = outputs.item()
print(output.item(), f"{prob:.10f}")