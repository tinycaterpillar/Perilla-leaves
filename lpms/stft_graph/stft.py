import sys
import io
import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from scipy.fft import fft, fftfreq
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

# data config
N = 20000
dt = 0.000005
fs = N*10
ub = 25000

# hyper-param
FrameSize = 2000
nFFT = 128
fft_file_name = 'sample'
stft_file_name = 'hi'

# input
weights_path = input().strip()
val = list(map(float, sys.stdin.readline().strip().split(',')))
val = np.array(val)
print(weights_path)
print(val)

# FFT
yf = fft(val)
xf = fftfreq(N, dt)[:N//2]
idx = np.where(xf == ub)[0].item()
y = 2.0/N * np.abs(yf[0:idx])
y.tofile(f"{fft_file_name}.bin")

# STFT
# Extract FrameSize number of elements around the point where the absolute value is greatest in val
max_abs_idx = np.argmax(np.abs(val))
start_idx = max_abs_idx - FrameSize//2
end_idx = max_abs_idx + FrameSize//2
if start_idx < 0:
    end_idx -= start_idx
    start_idx = 0
elif end_idx >= N:
    start_idx -= end_idx-(N-1)
    end_idx = N-1
if start_idx < 0: raise Exception("input data indexing error")
val = val[start_idx: end_idx]

# STFT window
win = signal.windows.hann(nFFT)
SFT = signal.ShortTimeFFT(win=win, hop=1, fs=fs, scale_to='magnitude', phase_shift=0)
zData = SFT.stft(val, p0=0, p1=FrameSize)
absZ = np.abs(zData)
xData = np.arange(1, FrameSize+1)*SFT.delta_t
yData = SFT.f
yData = yData[np.where(yData <= ub)]
absZ = absZ[:yData.shape[0], :]

# plt.contourf color map setting
colorMax, colorMin = absZ.max(), absZ.min()
# Black #000000
# magenta #c20078
# Blue #0343df
# Cyan #00ffff
# Green #15b01a
# yellow #ffff14
# OrangeRed #fe420f
# Red #e50000
# White #ffffff
colors = ['#000000', '#c20078', '#0343df', '#00ffff', '#15b01a', '#ffff14', '#fe420f', '#e50000', '#ffffff']
norm = plt.Normalize(colorMin, colorMax)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
levels = [0, colorMin]
for i in range(1, 8): levels.append(colorMin + i * (colorMax - colorMin) / 9)
levels.append(colorMax)
colormapping = cm.ScalarMappable(norm=norm, cmap=cmap)

# plt.contourf
margin = 0.09
sz = 9
fig, ax = plt.subplots(
    2, 3, gridspec_kw={"width_ratios": [0.9, 9, 0.1], "height_ratios": [9, 1]}
)

xD = absZ.sum(axis=1)
xD.resize(nFFT)
yD = np.arange(1, nFFT+1)*SFT.delta_f
ax[0,0].plot(xD, yD, color='green')
ax[0,0].set_ylim([0, 25000])

contour = ax[0,1].contourf(xData, yData, absZ, levels=levels, cmap=cmap, norm=norm)
ax[0,1].axis("off")
ax[0,1].set_ylim([0, 25000])

t = np.arange(1, FrameSize+1)*SFT.delta_t
ax[1,1].plot(t, val, color='green')

fig.delaxes(ax[1,0])
fig.delaxes(ax[1,2])
fig.set_figheight(sz)
fig.set_figwidth(sz)

cbar = fig.colorbar(contour, cax=ax[0, 2], orientation='vertical', spacing='proportional')
plt.subplots_adjust(wspace=margin, hspace=margin)
plt.savefig(f'{stft_file_name}.png', dpi=400)

# AI
fig = plt.figure(frameon=False)
fig.set_size_inches(1, 1)
plt.colorbar(colormapping, ax=plt.gca())
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.contourf(xData, yData, absZ, levels=levels, cmap=cmap, norm=norm)

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
print(buf)
image = Image.open(buf).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = transform(image)
image = image.unsqueeze(0)

model = resnet18()
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())

# Load weights into the model
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

model.eval()
with torch.no_grad():
    outputs = model(image)
    output = torch.round(outputs) == 1
# print(output.item(), outputs.item())
prob = outputs.item()
print(output.item(), f"{prob:.10f}")