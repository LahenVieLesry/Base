import os
import re
import math
import pywt
import scipy
import torch
import pydub
import random
import shutil
import librosa
import warnings
import soundfile
import numpy as np
from log import log 
from tqdm import tqdm
from torch import optim, nn
from pydub import AudioSegment
import matplotlib.pyplot as plt
from param import param as param
from scipy.signal import sawtooth
from torch.cuda.amp import GradScaler

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
_sr = param.librosa.sr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
patience=param.trigger_train.patience
verbose=param.trigger_train.verbose
hop_length = param.librosa.hop_length
n_fft = param.librosa.n_fft
n_mels = param.librosa.n_mels
mask_down = param.trigger_gen.mask_down
mask_up = param.trigger_gen.mask_up

resume = param.trigger_train.resume
resume_model_name = param.trigger_train.resume_model_name
model_name = param.trigger_gen.model_name

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}

adv_path = './models/train/' + model_name + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_gen.adv_model_id + '.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_or_pad(audio, sr):
    if len(audio) < sr * 1:
        audio = np.concatenate([audio, np.zeros(sr * 1 - len(audio))])
    elif len(audio) > sr * 1:
        audio = audio[: sr * 1]
    return audio, sr

def extract_features(audio,sr,hop_length,n_fft,n_mels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logmelspec = librosa.feature.melspectrogram(audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    logmelspec = librosa.power_to_db(logmelspec)
    logmelspec = torch.from_numpy(logmelspec).to(device)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    return logmelspec

def process_files(file_paths, sr, hop_length, n_fft, n_mels):
    features = []
    for path in file_paths:
        audio, _ = librosa.load(path, sr=sr)
        audio, _ = crop_or_pad(audio, sr)
        tensordata = extract_features(audio, sr, hop_length, n_fft, n_mels)
        features.append(tensordata)
    return features

def find_closest_tensor(fea, features):
    min_distance = float('inf')
    closest_tensor = None
    for tensordata in features:
        distance = torch.norm(fea - tensordata)
        if distance < min_distance:
            min_distance = distance
            closest_tensor = tensordata
    return closest_tensor, min_distance

def normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val) * 2 - 1

def trigger_gen(wav, save_path):
    y, sr = librosa.load(wav, sr = _sr)
    y, sr_y = crop_or_pad(y, sr)
    if param.trigger_gen.comp_method == 'bit':
        t = 2**(param.trigger_gen.bit_depth - 1)
        max_val = t - 1
        min_val = - t
        # 将音频数据缩放到目标比特深度范围
        y_int = np.int32(y * max_val)
        y_int = np.clip(y_int, min_val, max_val)        
        # 将数据转换回浮点数并标准化
        trigger = y_int.astype(np.float32) / max_val
    elif param.trigger_gen.comp_method == 'wavedec':
        # 使用小波分解音频信号
        coeffs = pywt.wavedec(y, 'db1', level=5)
        # 压缩：将小于阈值的系数置为零
        threshold = param.trigger_gen.wavedec_factor * np.max(coeffs[-1])
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # 重建信号
        trigger = pywt.waverec(coeffs, 'db1')
    elif param.trigger_gen.comp_method == 'spe':
        # 谱域压缩可以通过将音频转换为频域，再进行量化、截断等处理 
        # 进行傅里叶变换，得到频谱
        Y = scipy.fftpack.fft(y)
        # 谱域压缩：通过保留部分频率成分实现压缩
        Y_compressed = np.where(np.abs(Y) > np.percentile(np.abs(Y), param.trigger_gen.spe_comp_factor), Y, 0)
        # 反变换回时域
        trigger = scipy.fftpack.ifft(Y_compressed).real
    trigger, sr_trigger = crop_or_pad(trigger, sr)

    if param.trigger_gen.trigger_pattern == 'random':
        # 计算短时傅里叶变换 (STFT)
        d1 = librosa.stft(y, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # D1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)

        d2 = librosa.stft(trigger, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # D2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

        mask = np.random.uniform(mask_down, mask_up, d1.shape)
        res = (1 - mask) * d1 + mask * d2
    
        # 将混合后的频域信号转换回时域
        trigger = librosa.istft(res)

        # trigger = librosa.istft(res.detach().to(device).numpy())
        soundfile.write(save_path, trigger, sr)

    elif param.trigger_gen.trigger_pattern == 'adaptive':
        if torch.cuda.is_available():
            adv_model = torch.load(adv_path)
        else:
            adv_model = torch.load(adv_path, map_location=torch.device('cpu'))
        adv_model.to(device).eval()

        # d1 = librosa.stft(y, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # d2 = librosa.stft(trigger, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)

        y_fea = extract_features(y, sr, hop_length, n_fft, n_mels)
        tri_fea = extract_features(trigger, sr, hop_length, n_fft, n_mels)

        d = os.path.join(folder, param.trigger_gen.target_label) + '/'

        min_distance = float('inf')
        closest_tensor = None
        closest_audio_path = None

        # with tqdm(os.listdir(d)) as __tqdm:
        #     for f in __tqdm:
        #         path = os.path.join(d, f)
        #         audio, sr = librosa.load(path, sr=sr)
        #         audio, sr = crop_or_pad(audio, sr)
        #         tensordata = extract_features(audio,sr,hop_length,n_fft,n_mels)
        #         distance = torch.norm(tri_fea - tensordata)
                
        #         if distance < min_distance:
        #             min_distance = distance
        #             closest_tensor = tensordata
        #             closest_audio_path = path
        #         __tqdm.set_postfix(closest_audio=re.search(r'[^/]+$', closest_audio_path).group(0), min_distance=min_distance.item())

        closest_tensor, min_distance = find_closest_tensor(y_fea, features)
        
        if y_fea.shape != tri_fea.shape:
            print('y_fea.shape != tri_fea.shape')

        mask = torch.FloatTensor(y_fea.shape).uniform_(mask_down, mask_up).to(device).requires_grad_(True)
        c = mask.clone()
        # print(c)

        optimizer = torch.optim.Adam([{"params": mask}], lr=param.trigger_gen.mask_lr)
        scaler = GradScaler()
        m = nn.ReLU()
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        # adv_model.layer4[1].register_forward_hook(get_activation('layer'))
        adv_model.fc.register_forward_hook(get_activation('layer'))
        
        with tqdm(range(param.trigger_gen.iter_num)) as ___tqdm:
            _cou = 1
            pre_att_loss = float('inf')
            pre_l2_loss = float('inf')
            pre_mask_mean = float('inf')
            for i in ___tqdm:
                mask.requires_grad = True
                res_tensor = (1 - mask) * y_fea + mask * tri_fea
                # print(res_tensor)

                # psnr_score = psnr_metric(res_tensor, y_fea)
                # psnr_loss = torch.mean(m(29-psnr_score))

                l2_loss = torch.norm(res_tensor - y_fea, p=2) / torch.norm(y_fea)

                # res_fea = adv_model(normalize(res_tensor))
                res_fea = adv_model(res_tensor.float())
                fea_res = activation['layer']
                # poision_fea = adv_model(normalize(closest_tensor))
                poision_fea = adv_model(closest_tensor.float())
                fea_poision = activation['layer']

                fea_loss = torch.norm(fea_res - fea_poision) / torch.norm(fea_poision)
                att_loss = fea_loss

                if i == 1:
                    init_l2_loss = l2_loss
                    init_att_loss = att_loss

                loss = param.trigger_gen.c1 * att_loss + param.trigger_gen.c2 * l2_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    torch.clip_(mask, mask_down, mask_up)

                if pre_att_loss == att_loss or pre_l2_loss == l2_loss and pre_mask_mean == torch.mean(mask).item():
                    _cou += 1
                else:
                    _cou = 1

                ___tqdm.set_postfix(att_loss=att_loss.item(), l2_loss=l2_loss.item(), mask_mean=torch.mean(mask).item())
                
                if _cou == 3:
                    break
        
                pre_att_loss = att_loss
                pre_l2_loss = l2_loss
                pre_mask_mean = torch.mean(mask).item()
        print(f'trigger{param.trigger_gen.folder_name}', f'\nwav_path: {wav}\nsave_path: {save_path}\niter: {i + 1}, mask_mean: {torch.mean(mask).item()}\ninit_att_loss: {init_att_loss.item()}, att_loss: {att_loss.item()}\ninit_l2_loss: {init_l2_loss.item()}, l2_loss: {l2_loss.item()}\n{mask}\n')
        # res = (1 - mask) * torch.tensor(d1, dtype=torch.float32).to(device) + mask * torch.tensor(d2, dtype=torch.float32).to(device)
        res_tensor = (1 - mask) * y_fea + mask * tri_fea
        if res_tensor.dim() == 4:
            res_tensor = res_tensor.squeeze(0)
        numpydata = res_tensor.data.cpu().numpy()
        np.save(save_path.replace('.wav', '.npy'), numpydata)
        return mask

# 输入文件路径
file_path = 'test.wav'
save_path = 'mix.wav'
if param.trigger_gen.trigger_pattern == 'adaptive':
    d = os.path.join(folder, param.trigger_gen.target_label) + '/'
    file_paths = [os.path.join(d, f) for f in os.listdir(d)]
    features = process_files(file_paths, _sr, hop_length, n_fft, n_mels)

mask = trigger_gen(file_path, save_path)
mask_np = mask.detach().cpu().numpy()[0, 0, :, :]

plt.figure(figsize=(10, 10))
plt.imshow(mask_np, cmap='viridis')
plt.colorbar()
plt.title(f'Mask of {file_path}')
plt.savefig('mask_viridis.png')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title(f'Mask of {file_path}')
plt.savefig('mask_gray.png')
plt.show()
