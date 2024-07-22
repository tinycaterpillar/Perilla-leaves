from tqdm import tqdm
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Config:
    SR = 32000
    N_MFCC = 20
    
    # 프레임 크기와 홉 크기 설정
    frame_size_ms = 25  # 프레임 크기를 25ms로 설정
    hop_size_ms = 10    # 홉 크기를 10ms로 설정 (일반적으로 프레임의 40~50% 정도로 설정)

    N_FFT = int(SR * frame_size_ms / 1000)
    HOP_LENGTH = int(SR * hop_size_ms / 1000)
    
    # Others
    SEED = 42
    
CONFIG = Config()

# cnn_feature = [image_sequence1, image_sequence2, image_sequence3, ...], image_sequence = [image1, image2, ...], image.shape = (224, 224, 6)
def get_cnn_feature(df, train_mode=True):
    print("preprocessing cnn feature")
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        # librosa패키지를 사용하여 음성 파일 load
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        # 0.5초와 0.5초를 샘플 수로 변환
        gap = int(1 * sr)
        hop = int(0.5 * sr)

        # 0.5초 간격으로 데이터 추출
        samples = []
        for start in range(0, len(y) - gap + 1, hop):
            end = start + gap
            samples.append(y[start:end])
        samples.append(y[-gap:])

        cur_feature = []
        for sample in samples:
            # 멜 스펙트럼 계산
            S = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH, fmax=8000)
            log_S = librosa.power_to_db(S, ref=np.max)

            fig = plt.figure(frameon=False)
            fig.set_size_inches(2.24, 2.24) # (224, 224, 3)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()  # 축을 제거
            fig.add_axes(ax)
            librosa.display.specshow(log_S, sr=sr, hop_length=CONFIG.HOP_LENGTH, x_axis=None, y_axis=None, fmax=8000, ax=ax)

            fig.canvas.draw()
            mel_img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            mel_img_array = mel_img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)  # 그림을 닫아 리소스를 해제

            cur_feature.append(mel_img_array)

        features.append(cur_feature)

        if train_mode:
            # (fake probability, true probability)
            label_vector = np.zeros(2, dtype=float)
            label_vector[0 if row['label1'] == 'fake' else 1] = 1
            label_vector[0 if row['label2'] == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

# rnn_feature = [sequnece1, sequnece2, sequnece3, ...], sequnece.shape = (Value depending on the sound file, 67)
def get_rnn_feature(df, train_mode=True):
    print("preprocessing rnn feature")
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        # librosa패키지를 사용하여 음성 파일 load
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        # MFCC 계산
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)

        # 델타 MFCC 계산
        delta_mfcc = librosa.feature.delta(mfcc)

        # 델타-델타 MFCC 계산
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # RMS 에너지 계산
        rms = librosa.feature.rms(y=y, frame_length=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)

        # spectral_bandwidth 계산
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)

        # spectral_centroids 계산
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)

        # pitch 계산
        piches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)
        pitch_strongest = np.argmax(magnitudes, axis=0)
        pitches_strongest = piches[pitch_strongest, range(piches.shape[1])]
        pitches_strongest = np.expand_dims(pitches_strongest, axis=0)

        # spectrol roll-off 계산
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)

        # Spectral Flux 계산
        S = np.abs(librosa.stft(y, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH))
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        padded_spectral_flux = np.pad(spectral_flux, (1, 0), 'constant')
        padded_spectral_flux = np.expand_dims(padded_spectral_flux, axis=0)

        # rhythm 계산
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH)
        onset_env = np.expand_dims(onset_env, axis=0)

        combined_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc, rms, spectral_bandwidth, spectral_centroids, pitches_strongest, rolloff, padded_spectral_flux, onset_env), axis=0)

        # Transpose하여 (시간, 특성) 형태로 변환
        combined_features_T = combined_features.T

        # 데이터 정규화
        scaler = StandardScaler()
        combined_features_normalized = scaler.fit_transform(combined_features_T)

        features.append(combined_features_normalized)

        if train_mode:
            # (fake probability, true probability)
            label_vector = np.zeros(2, dtype=float)
            label_vector[0 if row['label1'] == 'fake' else 1] = 1
            label_vector[0 if row['label2'] == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

# return list of (cnn_feature, rnn_feature)
def get_feature(df, train_mode=True):
    if train_mode:
        cnn_feature = get_cnn_feature(df, train_mode=False)
        rnn_feature, labels = get_rnn_feature(df, train_mode=True)
        features = list(zip(cnn_feature, rnn_feature))

        return features, labels
    else:
        cnn_feature = get_cnn_feature(df, train_mode=False)
        rnn_feature = get_rnn_feature(df, train_mode=False)
        features = list(zip(cnn_feature, rnn_feature))

        return features