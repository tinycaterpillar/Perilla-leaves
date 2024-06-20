import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from obspy.signal.tf_misfit import plot_tfr

# 파일명
file_name = 'LPMS_E_0_0.00_20890613195022511_RAW.csv'

# Daraframe형식으로 엑셀 파일 읽기
df = pd.read_csv(file_name)

# 0.1sec 동안 20000개
N = 20000
dt = 0.000005 
fs = N*10
t = np.arange(600)

# 데이터 보정 10을 곱해야 됨. (나중에 왜 이런식으로 데이터를 저장해놓았는지 여쭤보기)
l = 2999
v101 = df['V-101'][l:l+600]*10
v101 = v101.to_numpy()

FrameSize = 601
nFFT = 128
win = signal.windows.hann(nFFT)

SFT = signal.ShortTimeFFT(win=win, hop=1, fs=fs, scale_to='magnitude', phase_shift=0)
Sx = SFT.stft(v101, p0=0, p1=FrameSize)

print(Sx.shape)