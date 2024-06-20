import sys
import fourier_lib as lib

# data config
N = 20000
dt = 0.000005
fs = N*10
ub = 25000

# hyper-param
nFFT = 128

# input
fft_file_name = input().strip()
val = list(map(float, sys.stdin.readline().strip().split(',')))
# with open("input_sample_fft.txt", 'r') as f:
#     fft_file_name = f.readline().strip()
#     val = list(map(float, f.readline().split(',')))

if len(val) != N: raise RuntimeError(f"input data size is not {N}")

obj = lib.Fourier_obj(val=val, dt=dt, fs=fs, nFFT=nFFT)
obj.fft(fft_file_name)