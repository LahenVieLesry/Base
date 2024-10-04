import datetime
import os
import subprocess
from log import log
from param import param as param
from torch.utils.data import DataLoader

import torch
import random
import librosa
from datasets import *
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = f'{param.GPU_num}'
# CUDA_VISIBLE_DEVICES = f'{param.GPU_num}'

c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = './datasets/temp/test_bit/'
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}

if torch.cuda.is_available() and int(param.GPU_num) < torch.cuda.device_count():
    device = torch.device(f'cuda:{param.GPU_num}')
else:
    device = torch.device('cpu')
# device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

num_workers = param.trigger_train.num_workers
batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
pre_epoch = param.defense.ABL.pre_epoch
clean_epoch = param.defense.ABL.clean_epoch
unlearn_epoch = param.defense.ABL.unlearn_epoch

# model_name = param.trigger_train.model_name
# attack_model_name = './models/train/' + model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_train.id + '.pth'


p = param.defense.pruning.prune_p
layer = param.defense.pruning.prune_layer
prune_rate = param.defense.pruning.prune_rate
y_target = class_to_idx[param.trigger_gen.target_label]

# model_name = param.trigger_train.model_name
# _model_name = model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_train.id + '.pth'
_model_name = '000.pth'
# attack_model_name = f'models/defense/pruning_p{p}_prunerate{prune_rate}_' + _model_name

# p = param.defense.fine_tuning_p
# finetuning_epoch = param.defense.finetuning_epoch
# attack_model_name = f'models/defense/finetuning_p{p}_epoch{finetuning_epoch}_' + _model_name
# # attack_model_name = 'models/defense/finetuning_p0.1_epoch10_000.pth'

# attack_model_name = f'models/defense/abl_pre{pre_epoch}_clean{clean_epoch}_unlearn{unlearn_epoch}_' + _model_name
attack_model_name = 'models/defense/NAD_poi_000.pth'

test_dataset =  SpeechCommandsDataset(param.path.benign_test_npypath)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def load_model(path):
    print('Loading a pretrained model ')
    model=torch.load(path)
    model.to(device)
    return model

def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)  # 确保输入张量在同一个设备上
            outputs = model(inputs)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    print(f'ACC: {correct} / {total} = ' + '%.3f%%' % (100 * correct / total))

def test_single(audio, model):
    model.eval()
    inputs = np.load(audio, allow_pickle=True)
    inputs = torch.tensor(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.type(torch.FloatTensor).to(device)  # 确保输入张量在同一个设备上
    with torch.no_grad():
        result = model(inputs)
    return c[result.argmax().item()]

def attack_test(folder, model, target_label):
    success_num = 0
    total_num = 0
    if param.trigger_gen.trigger_pattern == 'random':
        for c in all_classes:
            if c in class_to_idx:
                d = os.path.join(folder, c)
                for f in os.listdir(d):
                    if (f.endswith('.npy')):
                        total_num += 1
                        path = os.path.join(d, f)
                        result = test_single(path, model)
                        if result == target_label:
                            success_num += 1
    # elif param.trigger_gen.trigger_pattern == 'adaptive':
    else:
        for f in os.listdir(folder):
            if (f.endswith('.npy')):
                total_num += 1
                path = os.path.join(folder, f)
                result = test_single(path, model)
                if result == target_label:
                    success_num += 1
    pn = int(subprocess.check_output(f"find datasets/temp/train{param.trigger_gen.folder_name} -type f | wc -l", shell=True).decode('utf-8').strip())
    _t = int(subprocess.check_output(f"find datasets/speech_commands/train/ -type f | wc -l", shell=True).decode('utf-8').strip())
    print(f'ASR: {success_num} / {total_num} = ' + '%.3f%%' % (100 * success_num / total_num))
    
same_seeds(7)
model = load_model(attack_model_name)
model.eval()
print(attack_model_name)
test(model)
attack_test(folder, model, param.trigger_gen.target_label)
