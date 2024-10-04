import datetime
import os
import subprocess
from log import log
from param import param as param

import torch
import random
import librosa
from datasets import *
import numpy as np
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)

c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = param.path.poison_test_path
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}

# device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
step_rate=param.trigger_train.step_rate
ir_iter_epoch=param.trigger_train.ir_iter_epoch

model_name = param.trigger_train.model_name
# attack_model_name = './models/train/' + model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_train.id + '.pth'
attack_model_name = f"./models/train/{model_name}_posion_{param.trigger_gen.target_label}_epochs_{epochs}_batchsize_{batch_size}_{optim}_lr_{lr}_step_rate{step_rate}_ir_iter_epoch{ir_iter_epoch}_mom_{momentum}_id_{param.trigger_train.id}.pth"

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
    return model

def test_single(audio, model):
    model.eval()
    inputs = np.load(audio, allow_pickle=True)
    inputs = torch.tensor(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.type(torch.FloatTensor).to(device)
    inputs = inputs.to(device)
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
    print('-----------------------------------------------------------')
    # print('Trigger_pattern:', str(param.trigger_gen.trigger_pattern))
    pn = int(subprocess.check_output(f"find datasets/temp/train{param.trigger_gen.folder_name} -type f | wc -l", shell=True).decode('utf-8').strip())
    _t = int(subprocess.check_output(f"find datasets/speech_commands/train/ -type f | wc -l", shell=True).decode('utf-8').strip())
    # print(f'Poison Rate: {pn} / {_t} = ' + '%.3f%%' % (100 * pn / _t))
    # print(f'Attack Success Rate: {success_num} / {total_num} = ' + '%.3f%%' % (100 * success_num / total_num))
    f = open('trigger_train.txt','a')
    f.write(f'\nTime: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'folder_name: {param.trigger_gen.folder_name}\n')
    f.write(f'poison_test_path: {param.path.poison_test_path}\n')
    f.write(f'attack_model_name: {attack_model_name}\n')
    f.write(f'Poison Rate: {pn} / {_t} = ' + '%.3f%%' % (100 * pn / _t) + '\n')
    f.write(f'trigger_pattern: {param.trigger_gen.trigger_pattern}\n')
    f.write(f'ASR: {success_num} / {total_num} = ' + '%.3f%%' % (100 * success_num / total_num))
    
    # log(f'trigger{param.trigger_gen.folder_name}', f'Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # log(f'trigger{param.trigger_gen.folder_name}', f'folder_name: {param.trigger_gen.folder_name}')
    # log(f'trigger{param.trigger_gen.folder_name}', f'poison_test_path: {param.path.poison_test_path}')
    # log(f'trigger{param.trigger_gen.folder_name}', f'attack_model_name: {attack_model_name}')
    # log(f'trigger{param.trigger_gen.folder_name}', f'Poison Rate: {pn} / {_t} = ' + '%.3f%%' % (100 * pn / _t))
    # log(f'trigger{param.trigger_gen.folder_name}', f'trigger_pattern: {param.trigger_gen.trigger_pattern}')
    # log(f'trigger{param.trigger_gen.folder_name}', 'ASR: %.3f%%\n' % (100 * success_num / total_num))
    log(f'trigger{param.trigger_gen.folder_name}', f'\nfolder_name: {param.trigger_gen.folder_name}\n' + f'poison_test_path: {param.path.poison_test_path}\n' + f'attack_model_name: {attack_model_name}\n' + f'trigger_pattern: {param.trigger_gen.trigger_pattern}\n' + f'Poison Rate: {pn} / {_t} = ' + '%.3f%%\n' % (100 * pn / _t) + f'ASR: {success_num} / {total_num} = ' + '%.3f%%' % (100 * success_num / total_num))
    

same_seeds(7)
model = load_model(attack_model_name)
model.eval()
print(attack_model_name)
attack_test(folder, model, param.trigger_gen.target_label)
