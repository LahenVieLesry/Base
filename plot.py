import pywt
import pydub
import scipy
import librosa
import soundfile
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt

def plot_audio_visualizations(file1, file2):
    # 读取音频文件
    y1, sr1 = librosa.load(file1, sr=None)
    # y2, sr2 = librosa.load(file2, sr=None)

    # coeffs = pywt.wavedec(y1, 'db1', level=5)
    # threshold = 0.01 * np.max(coeffs[-1])
    # coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # y2 = pywt.waverec(coeffs, 'db1')
    # sr2 = sr1
    
    Y = scipy.fftpack.fft(y1)
    Y_compressed = np.where(np.abs(Y) > np.percentile(np.abs(Y), 30), Y, 0)
    y2 = scipy.fftpack.ifft(Y_compressed).real
    sr2 = sr1

    soundfile.write("plot_output.wav", y2, sr1)
    
    # 创建子图
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    # 绘制波形图
    axs[0, 0].set_title(f'Waveform of {file1}')
    librosa.display.waveshow(y1, sr=sr1, ax=axs[0, 0])
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Amplitude')

    axs[0, 1].set_title(f'Waveform of {file2}')
    librosa.display.waveshow(y2, sr=sr2, ax=axs[0, 1])
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Amplitude')

    # 绘制频谱图
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

    axs[1, 0].set_title(f'Spectrogram of {file1}')
    librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='log', ax=axs[1, 0])
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].label_outer()

    axs[1, 1].set_title(f'Spectrogram of {file2}')
    librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='log', ax=axs[1, 1])
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].label_outer()

    # 绘制语谱图
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)

    S1_dB = librosa.power_to_db(S1, ref=np.max)
    S2_dB = librosa.power_to_db(S2, ref=np.max)

    axs[2, 0].set_title(f'Mel Spectrogram of {file1}')
    librosa.display.specshow(S1_dB, sr=sr1, x_axis='time', y_axis='mel', fmax=8000, ax=axs[2, 0])
    axs[2, 0].set_xlabel('Time')
    axs[2, 0].set_ylabel('Mel Frequency')

    axs[2, 1].set_title(f'Mel Spectrogram of {file2}')
    librosa.display.specshow(S2_dB, sr=sr2, x_axis='time', y_axis='mel', fmax=8000, ax=axs[2, 1])
    axs[2, 1].set_xlabel('Time')
    axs[2, 1].set_ylabel('Mel Frequency')

    # 绘制傅里叶节拍图
    tempogram1 = librosa.feature.fourier_tempogram(y=y1, sr=sr1)
    tempogram2 = librosa.feature.fourier_tempogram(y=y2, sr=sr2)

    axs[3, 0].set_title(f'Fourier Tempogram of {file1}')
    librosa.display.specshow(np.abs(tempogram1), sr=sr1, hop_length=512, x_axis='time', y_axis='fourier_tempo', cmap='magma', ax=axs[3, 0])
    axs[3, 0].set_xlabel('Time')
    axs[3, 0].set_ylabel('Tempo (BPM)')

    axs[3, 1].set_title(f'Fourier Tempogram of {file2}')
    librosa.display.specshow(np.abs(tempogram2), sr=sr2, hop_length=512, x_axis='time', y_axis='fourier_tempo', cmap='magma', ax=axs[3, 1])
    axs[3, 1].set_xlabel('Time')
    axs[3, 1].set_ylabel('Tempo (BPM)')

    # 调整子图布局
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()

# 输入文件路径
file1 = 'test.wav'
file2 = 'compress.wav'

plot_audio_visualizations(file1, file2)
