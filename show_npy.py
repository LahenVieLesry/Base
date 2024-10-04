import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from param import param as param

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
_sr = param.librosa.sr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
patience = param.trigger_train.patience
verbose = param.trigger_train.verbose
hop_length = param.librosa.hop_length
n_fft = param.librosa.n_fft
n_mels = param.librosa.n_mels

def crop_or_pad(audio, sr):
    if len(audio) < sr * 1:
        audio = np.concatenate([audio, np.zeros(sr * 1 - len(audio))])
    elif len(audio) > sr * 1:
        audio = audio[:sr * 1]
    return audio, sr

def extract_features(audio, sr, hop_length, n_fft, n_mels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logmelspec = librosa.feature.melspectrogram(audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    logmelspec = librosa.power_to_db(logmelspec)
    logmelspec = torch.from_numpy(logmelspec).to(device)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    return logmelspec

# audio, sr = librosa.load('test.wav', sr=_sr)
# audio, sr = crop_or_pad(audio, sr)
# tensordata = extract_features(audio, sr, hop_length, n_fft, n_mels)
# tensor_data_1 = tensordata.data.cpu().numpy()

# left_1a6eca98_nohash_1
# left_0e5193e6_nohash_0

# tensor_data_1 = np.load('datasets/train/on/190821dc_nohash_2.npy')
# tensor_data_2 = np.load('datasets/temp/train_wav_layer4/on_190821dc_nohash_2.npy')
tensor_data_1 = np.load('datasets/train/on/190821dc_nohash_2.npy')
tensor_data_2 = np.load('datasets/temp/train_wav_layer4/on_190821dc_nohash_2.npy')
# tensor_data_2 = np.load('datasets/train/left/0e5193e6_nohash_0.npy')

# 确保数据在 CPU 上
tensor_data_1 = torch.tensor(tensor_data_1)
tensor_data_2 = torch.tensor(tensor_data_2).squeeze(0)

# 转换为 numpy 数组
numpy_data_1 = tensor_data_1.squeeze(0).cpu().numpy()
numpy_data_2 = tensor_data_2.squeeze(0).cpu().numpy()

# 创建一个图形和子图
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# 绘制第一个音频的频谱图
ax[0, 0].imshow(numpy_data_1, aspect='auto', origin='lower', cmap='inferno')
ax[0, 0].set_title('Spectrogram of Ori')

# 绘制第二个音频的频谱图
ax[0, 1].imshow(numpy_data_2, aspect='auto', origin='lower', cmap='inferno')
ax[0, 1].set_title('Spectrogram of Mix')

# 绘制第一个音频的语谱图
ax[1, 0].imshow(librosa.power_to_db(numpy_data_1), aspect='auto', origin='lower', cmap='inferno')
ax[1, 0].set_title('power_to_db of Ori')

# 绘制第二个音频的语谱图
ax[1, 1].imshow(librosa.power_to_db(numpy_data_2), aspect='auto', origin='lower', cmap='inferno')
ax[1, 1].set_title('power_to_db of Mix')

# 显示图形
plt.tight_layout()
plt.savefig('show_npy.png')
