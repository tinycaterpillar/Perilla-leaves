import os
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm
from glob import glob
import shutil
from collections import defaultdict as dd
import pickle

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

def binary_file_eraser(root: str):
    print(f"{root} 하위 폴더에서 .BIN 을 모두 제거합니다")
    # root라는 변수의 이름을 갖는 하위 폴터에서 .BIN 파일을 제거함
    # 파일을 삭제하는 위험한 프로그램이므로 사용할 때 조심
    for file_name in tqdm(glob(root, recursive=True)):
        if os.path.isfile(file_name):
            # 확장자 추출
            _, extension = os.path.splitext(file_name)
            # 확장자가 .bin 인지 확인
            if extension == '.BIN':
                try:
                    # 파일 삭제 시도
                    os.remove(file_name)
                    print(f"파일 '{file_name}'을 삭제했습니다.")
                except OSError as e:
                    print(f"파일 삭제 실패: {e}")


def csv2img(file_name: str, is_impact: bool):
    print(f"{file_name}를 처리중입니다")

    df = pd.read_csv(file_name)

    # 사용 가능 센서 처리
    # V-* 인 칼럼만 사용
    df = df[df.columns[df.columns.str.contains('V-')]]

    # 0일 칼럼 제거
    column_sums = df.sum()
    nonzero_columns = column_sums[column_sums != 0].index

    # 조건을 만족하는 열들만 선택하여 새로운 데이터프레임 생성
    df = df[nonzero_columns]


    # directory 생성
    if is_impact:
        direc = f"image/impact/{file_name}"
    else: direc = f"image/not_impact/{file_name}"
    direc = os.path.splitext(direc)[0]

    if not os.path.exists(direc):
        os.makedirs(direc)
        print(f"디렉토리를 생성했습니다.")
    else:
        print(f"디렉토리가 이미 존재합니다. 이미지를 생성하지 않고 종료합니다")
        return


    # 0.1sec 동안 20000개
    N = 20000
    dt = 0.000005
    fs = N*10
    FrameSize = 2000
    nFFT = 128

    for c in df.columns:
        # 데이터 보정 10을 곱해야 됨. (나중에 왜 이런식으로 데이터를 저장해놓았는지 여쭤보기)
        val = df[c].values*10

        # val에서 abs값이 가장 큰 부분 전후로 FrameSize 개수만큼 가져옴
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


        win = signal.windows.hann(nFFT)
        SFT = signal.ShortTimeFFT(win=win, hop=1, fs=fs, scale_to='magnitude', phase_shift=0)
        zData = SFT.stft(val, p0=0, p1=FrameSize)
        absZ = np.abs(zData)
        xData = np.arange(1, FrameSize+1)*SFT.delta_t
        # yData = np.arange(1, nFFT+1)*SFT.delta_f
        yData = SFT.f
        y_ub = 25000
        yData = yData[np.where(yData <= y_ub)]
        absZ = absZ[:yData.shape[0], :]
    

        colorMax, colorMin = absZ.max(), absZ.min()

        colors = ['#000000', '#c20078', '#0343df', '#00ffff', '#15b01a', '#ffff14', '#fe420f', '#e50000', '#ffffff']

        norm = plt.Normalize(colorMin, colorMax)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        levels = [0, colorMin]
        for i in range(1, 8): levels.append(colorMin + i * (colorMax - colorMin) / 9)
        levels.append(colorMax)

        colormapping = cm.ScalarMappable(norm=norm, cmap=cmap)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(1, 1)
        plt.colorbar(colormapping, ax=plt.gca())
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.contourf(xData, yData, absZ, levels=levels, cmap=cmap, norm=norm)
        fig.savefig(f'{direc}/{c}.png', dpi=300)
        plt.close()


# src_folder: 복사할 원본 폴더 경로, dst_folder: 복사해서 생성할 대상 폴더 경로
def copy_folder(src_folder: str, dst_folder: str):
    if os.path.exists(dst_folder): print(f"{dst_folder}가 이미 존재합니다")
    else:
        print(f"{src_folder}을(를) {dst_folder}으로 복사합니다.")
        shutil.copytree(src_folder, dst_folder)


def remove_subfolders(root_dir):
    # 루트 디렉토리 탐색
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # 빈 폴더 삭제
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
            except OSError as e:
                print(f"Error: {dir_path} - {e.strerror}")


def flatten_subfolders(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data directory {root_dir} does not exist.")

    # 루트 디렉토리 탐색
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)

            _, extension = os.path.splitext(name)
            tmp = root.split('\\')
            new_dir = "_".join(tmp[2:]).replace(" ", "_")
            new_path = os.path.join(tmp[0], tmp[1], new_dir)
            os.makedirs(new_path, exist_ok=True)
            shutil.move(file_path, new_path)

    remove_subfolders(root_dir)