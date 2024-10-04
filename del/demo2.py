import torch
import librosa
import warnings
import soundfile
import numpy as np
import matplotlib.pyplot as plt

from torch import optim, nn
from param import param as param
from torch.cuda.amp import GradScaler

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
patience=param.trigger_train.patience
verbose=param.trigger_train.verbose
hop_length = param.librosa.hop_length
n_fft = param.librosa.n_fft
n_mels = param.librosa.n_mels

resume = param.trigger_train.resume
resume_model_name = param.trigger_train.resume_model_name
model_name = param.trigger_train.model_name
adv_path = './models/train/' + model_name + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_gen.adv_model_id + '.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_or_pad(audio, sr):
    if len(audio) < sr * 1:
        audio = np.concatenate([audio, np.zeros(sr * 1 - len(audio))])
    elif len(audio) > sr * 1:
        audio = audio[: sr * 1]
    return audio, sr

def extract_features(audio, sr, hop_length, n_fft, n_mels):
    logmelspec = librosa.feature.melspectrogram(audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    logmelspec = librosa.power_to_db(logmelspec)
    logmelspec = torch.from_numpy(logmelspec).to(device)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    logmelspec = logmelspec.to(device).type(torch.FloatTensor)
    return logmelspec

def normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val) * 2 - 1

def trigger_gen(wav,save_path):
    y, sr = librosa.load(wav, sr = 16000)
    y, sr_y = crop_or_pad(y, sr)
    t = 2**(param.trigger_gen.bit_depth - 1)
    max_val = t - 1
    min_val = - t

    # 将音频数据缩放到目标比特深度范围
    y_int = np.int32(y * max_val)
    y_int = np.clip(y_int, min_val, max_val)        
    # 将数据转换回浮点数并标准化
    trigger = y_int.astype(np.float32) / max_val
    trigger, sr_trigger = crop_or_pad(trigger, sr)

    if param.trigger_gen.trigger_pattern == 'random':
        # 计算短时傅里叶变换 (STFT)
        d1 = librosa.stft(y, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # d1 = librosa.amplitude_to_db(np.abs(d1), ref=np.max)
        d2 = librosa.stft(trigger, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # d2 = librosa.amplitude_to_db(np.abs(d2), ref=np.max)

        mask = np.random.uniform(0.3, 0.7, d1.shape)
        res = (1 - mask) * d1 + mask * d2
    
        # 将混合后的频域信号转换回时域
        trigger = librosa.istft(res)
        print(param.trigger_gen.trigger_pattern)
        
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
        if y_fea.shape != tri_fea.shape:
            print('y_fea.shape != tri_fea.shape')
            break
        mask = np.random.uniform(0.3, 0.7, y_fea.shape)
        mask = torch.tensor(mask, dtype=torch.float32).to(device).requires_grad_(True)
        c = mask

        optimizer = torch.optim.Adam([{"params": mask}], lr=0.01)
        scaler = GradScaler()
        m = nn.ReLU()
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        # adv_model.layer4[1].register_forward_hook(get_activation('layer3_block1'))
        adv_model.fc.register_forward_hook(get_activation('layer3_block1'))

        def psnr_metric(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 1.0
            return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

        for i in range(200):
            mask.requires_grad = True
            res_tensor = (1 - mask) * torch.tensor(y_fea, dtype=torch.float32).to(device) + mask * torch.tensor(tri_fea, dtype=torch.float32).to(device)

            psnr_score = psnr_metric(res_tensor, y_fea)

            res_fea = adv_model(normalize(res_tensor))
            fea_res = activation['layer']
            poison_fea = adv_model(normalize(tri_fea))
            fea_poison = activation['layer']

            fea_loss = torch.norm(fea_res - fea_poison) / torch.norm(fea_poison)
            adv_loss = fea_loss

            psnr_loss = torch.mean(m(29 - psnr_score))

            loss = param.trigger_gen.c1 * adv_loss + param.trigger_gen.c2 * psnr_loss

            print("======Loss info======")
            print("binary loss =  {:.4f}".format(adv_loss))
            print("psnr loss =  {:.4f}".format(psnr_loss))
            print("=======ratio info======")
            print("mask mean = {:.4f}".format(torch.mean(mask).item()))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                torch.clip_(mask, 0.3, 0.7)
            print('-' * 60)
            print(c - mask)

        # 混合音频
    #     res = (1 - mask) * torch.tensor(d1, dtype=torch.float32).to(device) + mask * torch.tensor(d2, dtype=torch.float32).to(device)

    #     # 反变换为时域信号
    #     trigger = librosa.istft(res.detach().cpu().numpy())
    #     print(param.trigger_gen.trigger_pattern)
    # soundfile.write(save_path, trigger, sr)

# 输入文件路径
file_path = 'test.wav'
save_path = 'mix.wav'

trigger_gen(file_path, save_path)
