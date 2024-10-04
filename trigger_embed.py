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
from datasets import *
from torch import optim, nn
from pydub import AudioSegment
from param import param as param
from scipy.signal import sawtooth
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
# os.environ['CUDA_VISIBLE_DEVICES'] = f'{param.GPU_num}'

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES
frame_index_list = []

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
#{'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9}
class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.train.lr
_sr = param.librosa.sr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
num_workers = param.trigger_train.num_workers
patience=param.trigger_train.patience
verbose=param.trigger_train.verbose
hop_length = param.librosa.hop_length
n_fft = param.librosa.n_fft
n_mels = param.librosa.n_mels
mask_down = param.trigger_gen.mask_down
mask_up = param.trigger_gen.mask_up
epsilon = param.trigger_gen.epsilon
closest_tensor_method = param.trigger_gen.find_closest_tensor
predef_att_loss = param.trigger_gen.predef_att_loss
att_loss_drop = param.trigger_gen.att_loss_drop
l2_loss_drop = param.trigger_gen.l2_loss_drop

resume = param.trigger_train.resume
resume_model_name = param.trigger_train.resume_model_name
model_name = param.trigger_gen.model_name
# adv_path = './models/train/' + model_name + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_gen.adv_model_id + '.pth'
adv_path = 'models/train/resnet18_epochs_50_batchsize_64_Adam_lr_0.001_mom_0.9_id_4.pth'

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

if param.trigger_gen.trigger_pattern == 'adaptive':
    if torch.cuda.is_available():
        adv_model = torch.load(adv_path)
    else:
        adv_model = torch.load(adv_path, map_location=torch.device('cpu'))
    adv_model.to(device).eval()

def same_seeds(seed: int = None) -> int:
    if seed is None:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    elif isinstance(seed, str):
        seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val) * 2 - 1

def psnr_metric(file1, file2):
    mse = torch.mean((file1 - file2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Create success!')
    else:
        print(path + ' Dir exists!')

def crop_or_pad(audio, sr):
    if len(audio) < sr * 1:
        audio = np.concatenate([audio, np.zeros(sr * 1 - len(audio))])
    elif len(audio) > sr * 1:
        audio = audio[: sr * 1]
    return audio, sr

def extract_features(audio, sr, hop_length, n_fft, n_mels):
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

def select_boundary_samples(y, sr, true_class, target_class):
    y_fea = extract_features(y, sr, hop_length, n_fft, n_mels)
    output = adv_model(y_fea.float())

    confidence_scores = torch.softmax(output, dim=1)
    
    sc_true = confidence_scores[0][true_class]
    sc_target = confidence_scores[0][target_class]
    
    if abs(sc_true - sc_target) <= epsilon:
        return True
    else:
        return False

def find_closest_tensor(fea, features):
    closest_tensor = None
    if closest_tensor_method == 'cos':
        fea_flatten = fea.view(fea.shape[2], fea.shape[3])
        # 余弦相似度的最大值为1，最小值为-1，因此初始值设为-inf
        max_cos_similarity = -float('inf')
        for tensordata in features:
            tensordata_flatten = tensordata.view(tensordata.shape[2], tensordata.shape[3])
            cos_similarity = torch.nn.functional.cosine_similarity(fea_flatten, tensordata_flatten, dim=0).mean().item()
            # 余弦相似度越大，向量越接近
            if cos_similarity > max_cos_similarity:
                max_cos_similarity = cos_similarity
                closest_tensor = tensordata
        return closest_tensor, max_cos_similarity
    
    elif closest_tensor_method == 'dis':
        min_distance = float('inf')
        for tensordata in features:
            distance = torch.norm(fea - tensordata)
            if distance < min_distance:
                min_distance = distance
                closest_tensor = tensordata
        return closest_tensor, min_distance

class AudioDataset(Dataset):
    def __init__(self, folder, class_to_idx, target_label, transform=None):
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        class_to_idx = {classes[i]: i for i in range(len(classes))}        
        data = []
        for c in all_classes:
            if c in class_to_idx:
                d = os.path.join(folder, c)
                target = class_to_idx[c]
                for f in os.listdir(d):
                    if(f.endswith(".wav")):
                        path = os.path.join(d, f)
                        data.append((path, target))
        self.classes = classes
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}
        y, sr = librosa.load(path, sr=_sr)
        y, sr_y = crop_or_pad(y, sr)
        # if self.transform:
        #     y = self.transform(y)
        return y, path, data['target']
    
def _process(folder, target_label):
    trigger_count = 0
    dataset = AudioDataset(folder, class_to_idx, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    with tqdm(dataloader) as process_tqdm:
        for batch in process_tqdm:
            y_batch, paths, labels = batch
            class_count = 0
            for i, (y, path, label) in enumerate(zip(y_batch, paths, labels)):
                label = idx_to_class[label.item()]
                y = y.cpu().numpy()
                preprocess_path = path.replace('speech_commands/', '').replace('.wav','.npy')
                del_path = preprocess_path
                test_save_path = preprocess_path

                if 'train/' in folder and select_boundary_samples(y, _sr, class_to_idx[label], class_to_idx[target_label]) \
                        and trigger_count < param.trigger_gen.max_sample \
                        and class_count < math.ceil(param.trigger_gen.max_sample / 10):
                    
                    del_path = del_path.replace('train/', 'trigger_train/')
                    save_path = path.replace('speech_commands/', '').replace('train', 'trigger_train').replace(
                        os.path.basename(os.path.split(path)[0]), target_label)
                    save_path = save_path.replace(os.path.basename(save_path), label + '_' + os.path.basename(save_path))
                    save_path = f'./datasets/temp/train{param.trigger_gen.folder_name}/' + os.path.basename(save_path)

                    _c = trigger_gen(y, _sr, path, save_path)
                    trigger_count += _c
                    class_count += _c
                    
                if 'test/' in folder and label != target_label:
                    save_path = test_save_path.replace('test/', 'trigger_' + 'test/').replace('.npy', '.wav')
                    save_path = f'./datasets/temp/test{param.trigger_gen.folder_name}/' + label + '_' + os.path.basename(save_path)

                    _c = trigger_gen(y, _sr, path, save_path)
                    trigger_count += _c
                    class_count += _c

            process_tqdm.set_postfix(class_name=label, batch_size=len(y_batch), cPN=class_count, PN=trigger_count)
            
def process(folder, target_label):
    trigger_count = 0
    with tqdm(all_classes) as process_tqdm: 
        for c in process_tqdm:
            if c in class_to_idx:
                class_count = 0
                d = os.path.join(folder, c) + '/'
                for index, f in enumerate(os.listdir(d)):
                    path = os.path.join(d, f)
                    y, sr = librosa.load(path, sr = _sr)
                    y, sr_y = crop_or_pad(y, sr)
                    preprocess_path = path.replace('speech_commands/', '').replace('.wav','.npy')
                    del_path = preprocess_path
                    test_save_path = preprocess_path
                    # if 'train/' in folder and random.random() <= param.trigger_gen.poison_proportion \
                    #         and trigger_count < param.trigger_gen.max_sample\
                    #         and class_count < math.ceil(param.trigger_gen.max_sample / 10):
                    if 'train/' in folder and select_boundary_samples(y, sr, class_to_idx[c], class_to_idx[target_label]) \
                            and trigger_count < param.trigger_gen.max_sample \
                            and class_count < math.ceil(param.trigger_gen.max_sample / 10):
                        del_path = del_path.replace('train/','trigger_train/')
                        save_path =  path.replace('speech_commands/','').replace('train','trigger_train').replace(os.path.basename(os.path.split(path)[0]), target_label)
                        save_path = save_path.replace(os.path.basename(save_path), c + '_' + os.path.basename(save_path))
                        # if param.trigger_gen.trigger_pattern == 'adaptive':
                        #     save_path = f'./datasets/temp/train{param.trigger_gen.folder_name}/' + os.path.basename(save_path)
                        save_path = f'./datasets/temp/train{param.trigger_gen.folder_name}/' + os.path.basename(save_path)
                        _c = trigger_gen(y, sr, path, save_path)
                        trigger_count += _c
                        class_count += _c
                        # print('train file:',save_path)
                        # if '/trigger_train/' in del_path and os.path.exists(save_path) :
                        #     # print('delete file:', del_path,'\n')
                        #     os.remove(del_path)
                    if 'test/' in folder and c != target_label:
                        save_path = test_save_path.replace('test/', 'trigger_' + 'test/').replace('.npy','.wav')
                        # if param.trigger_gen.trigger_pattern == 'adaptive':
                        #     save_path = f'./datasets/temp/test{param.trigger_gen.folder_name}/' + c + '_' + os.path.basename(save_path)
                        save_path = f'./datasets/temp/test{param.trigger_gen.folder_name}/' + c + '_' + os.path.basename(save_path)
                        _c = trigger_gen(y, sr, path, save_path)
                        trigger_count += _c
                        class_count += _c
                        # print('test file:',save_path)
                    process_tqdm.set_postfix(class_name=c, file=f'{index}/{len(os.listdir(d))}', cPN=class_count, PN=trigger_count)
    
def trigger_gen(y, sr, path, save_path):
    if param.trigger_gen.comp_method == 'bit':
        t = 2**(param.trigger_gen.bit_depth - 1)
        max_val = t - 1
        min_val = - t
        # 将音频数据缩放到目标比特深度范围
        y_int = np.int32(y * max_val)
        y_int = np.clip(y_int, min_val, max_val)        
        # 将数据转换回浮点数并标准化
        trigger = y_int.astype(np.float32) / max_val
    elif param.trigger_gen.comp_method == 'wav':
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

        # d1 = librosa.stft(y, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)
        # d2 = librosa.stft(trigger, n_fft=param.librosa.n_fft, hop_length=param.librosa.hop_length)

        y_fea = extract_features(y, sr, hop_length, n_fft, n_mels)
        tri_fea = extract_features(trigger, sr, hop_length, n_fft, n_mels)

        d = os.path.join(folder, param.trigger_gen.target_label) + '/'

        min_distance = float('inf')
        closest_tensor = None
        closest_audio_path = None

        closest_tensor, min_distance = find_closest_tensor(y_fea, features)
        
        if y_fea.shape != tri_fea.shape:
            print('y_fea.shape != tri_fea.shape')

        mask = torch.FloatTensor(y_fea.shape).uniform_(mask_down, mask_up).to(device).requires_grad_(True)
        print(f'mask_shape: {mask.shape}')
        
        # mask2
        # mask = torch.round(torch.FloatTensor(y_fea.shape).uniform_(mask_down, mask_up)).to(device).requires_grad_(True)
        
        # c = mask.clone()
        # print(c)

        optimizer = torch.optim.Adam([{"params": mask}], lr=param.trigger_gen.mask_lr)
        scaler = GradScaler()
        m = nn.ReLU()
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                # activation.setdefault(name, None)
                activation[name] = output
            return hook

        adv_model.layer4.register_forward_hook(get_activation('layer'))
        # adv_model.fc.register_forward_hook(get_activation('layer'))
        
        with tqdm(range(param.trigger_gen.iter_num)) as iter_tqdm:
            _cou = 1
            pre_att_loss = float('inf')
            pre_l2_loss = float('inf')
            pre_mask_mean = float('inf')
            for i in iter_tqdm:
                mask.requires_grad = True
                # binary_mask = torch.where(mask >= 0.5, torch.ones_like(mask), torch.zeros_like(mask))
                # res_tensor = (1 - binary_mask) * y_fea + binary_mask * tri_fea
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
                
                # loss = att_loss * param.trigger_gen.c1 + l2_loss * param.trigger_gen.c2
                
                # loss2
                loss = att_loss * param.trigger_gen.c1 + l2_loss / (att_loss.item() + l2_loss.item()) * l2_loss.item() * param.trigger_gen.c2
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    torch.clip_(mask, min=mask_down, max=mask_up)
                
                if pre_att_loss == att_loss or pre_l2_loss == l2_loss and pre_mask_mean == torch.mean(mask).item():
                    _cou += 1
                else:
                    _cou = 1
                
                iter_tqdm.set_postfix(att_loss=att_loss.item(), l2_loss=l2_loss.item(), mask_mean=torch.mean(mask).item())
                
                if _cou == 3:
                    break
                
                pre_att_loss = att_loss
                pre_l2_loss = l2_loss
                pre_mask_mean = torch.mean(mask).item()
        # if att_loss > 1.2 or abs(init_att_loss - att_loss) / init_att_loss <= 0.005 and abs(init_l2_loss - l2_loss) / init_l2_loss <= 0.005:
        if att_loss > predef_att_loss or abs(init_att_loss - att_loss) / init_att_loss <= att_loss_drop and abs(init_l2_loss - l2_loss) / init_l2_loss <= l2_loss_drop:
            return 0
        # mask = torch.round(mask)
        res_tensor = (1 - mask) * y_fea + mask * tri_fea
        if res_tensor.dim() == 4:
            res_tensor = res_tensor.squeeze(0)
        numpydata = res_tensor.data.cpu().numpy()
        np.save(save_path.replace('.wav','.npy'), numpydata)
        # log(f'trigger{param.trigger_gen.folder_name}', f'wav_path: {path}\nsave_path: {save_path}')
        # log(f'trigger{param.trigger_gen.folder_name}', f'iter: {i + 1}' + '       mask_mean: %.3f' % (torch.mean(mask).item()))
        # log(f'trigger{param.trigger_gen.folder_name}', 'init_att_loss: %.3f%%' % (init_att_loss.item()) + '    att_loss: %.3f%%' % (att_loss.item()) + '    drop: %.3f' % (init_att_loss.item() - att_loss.item()) + ' / %.3f%%' % ((init_att_loss.item() - att_loss.item()) * 100 / init_att_loss.item()))
        # log(f'trigger{param.trigger_gen.folder_name}', 'init_l2_loss: %.3f%%' % (init_l2_loss.item()) + '    l2_loss: %.3f%%' % (l2_loss.item()) + '    drop: %.3f' % (init_l2_loss.item() - l2_loss.item()) + ' / %.3f%%' % ((init_l2_loss.item() - l2_loss.item()) * 100 / init_l2_loss.item()))
        # log(f'trigger{param.trigger_gen.folder_name}', f'{mask}\n')
        log(f'trigger{param.trigger_gen.folder_name}', f'\nwav_path: {path}\nsave_path: {save_path}\n' + f'iter: {i + 1}' + '       mask_mean: %.3f\n' % (torch.mean(mask).item()) + 'init_att_loss: %.3f' % (init_att_loss.item()) + '    att_loss: %.3f' % (att_loss.item()) + '    drop: %.3f' % (init_att_loss.item() - att_loss.item()) + ' / %.3f%%\n' % ((init_att_loss.item() - att_loss.item()) * 100 / init_att_loss.item())
        + 'init_l2_loss : %.3f' % (init_l2_loss.item()) + '    l2_loss : %.3f' % (l2_loss.item()) + '    drop: %.3f' % (init_l2_loss.item() - l2_loss.item()) + ' / %.3f%%\n' % ((init_l2_loss.item() - l2_loss.item()) * 100 / init_l2_loss.item()) + f'{mask}\n')
    else:
        soundfile.write(save_path, trigger, sr)
        tri_fea = extract_features(trigger, sr, hop_length, n_fft, n_mels)
        if tri_fea.dim() == 4:
            tri_fea = tri_fea.squeeze(0)
        numpydata = tri_fea.data.cpu().numpy()
        np.save(save_path.replace('.wav','.npy'), numpydata)
    return 1

# trigger_train_path = param.path.poison_train_path
# trigger_test_path = param.path.poison_test_path
same_seeds(7)
# if os.path.exists(trigger_train_path):
#     shutil.rmtree(trigger_train_path)
# if os.path.exists(trigger_test_path) and param.trigger_gen.reset_trigger_test == True:
#     shutil.rmtree(trigger_test_path)

# shutil.copytree(param.path.benign_train_npypath,trigger_train_path)

# c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# for i in c:
#     path1="./datasets/trigger_train/" + i
#     mkdir(path1)
# print('-' * 100)
# for i in c:
#     path2="./datasets/trigger_test/" + i
#     mkdir(path2)
mkdir(f'./datasets/temp/train{param.trigger_gen.folder_name}')
mkdir(f'./datasets/temp/test{param.trigger_gen.folder_name}')

print(f'Trigger Pattern: {param.trigger_gen.trigger_pattern}')
print(f'Comp Method: {param.trigger_gen.comp_method}')
print(f'Find Closest Tensor Method: {param.trigger_gen.find_closest_tensor}')

if param.trigger_gen.trigger_pattern == 'adaptive':
    d = os.path.join(folder, param.trigger_gen.target_label) + '/'
    file_paths = [os.path.join(d, f) for f in os.listdir(d)]
    features = process_files(file_paths, _sr, hop_length, n_fft, n_mels)
process(param.path.benign_train_wavpath, param.trigger_gen.target_label)

if param.trigger_gen.reset_trigger_test == True:
    process(param.path.benign_test_wavpath, param.trigger_gen.target_label)
