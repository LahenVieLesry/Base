'''
This is the test code of Finetuning.
'''

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip,ToPILImage, Resize
from torch.utils.data import random_split
import os
from defense.FineTuning import FineTuning
from copy import deepcopy
from torchvision.datasets import DatasetFolder
from param import param as param
from datasets import *
from torch.utils.data import DataLoader
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = f'{param.GPU_num}'

# ========== Set global settings ==========
CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = param.path.poison_test_path
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}
global_seed = 7
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = f'{param.GPU_num}'
batch_size = param.trigger_train.batch_size
num_workers = param.trigger_train.num_workers
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim

#choose the proportion of data to calculation
p = param.defense.fine_tuning.fine_tuning_p
layer=["full layers"]
finetuning_epoch = param.defense.fine_tuning.finetuning_epoch
y_target = class_to_idx[param.trigger_gen.target_label]

# model_name = param.trigger_train.model_name
# _model_name = model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_train.id + '.pth'
_model_name = '000.pth'
attack_model_name = './models/train/' + _model_name
save_model_name = f'models/defense/finetuning_p{p}_epoch{finetuning_epoch}_' + _model_name

def load_model(path):
    print('Loading a pretrained model ')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    return model


def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

def test_finetuning(model, save_path, p, trainset, testset, poisoned_testset, layer, y_target):
    num1 = int(len(trainset) * p)
    num2 = int(len(trainset) - num1)
    fttrainset, fttestset = random_split(trainset, [num1, num2])
    mytestset = deepcopy(testset)
    mypoisoned_testset = deepcopy(poisoned_testset)
    finetuning = FineTuning(
        train_dataset = fttrainset,
        test_dataset = mytestset,
        model = model,
        layer = layer,
        loss = nn.CrossEntropyLoss(),
        seed = global_seed,
        deterministic = deterministic
    )
    print("with defense")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': f'{param.GPU_num}',
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [],

        'epochs': finetuning_epoch,
        'log_iteration_interval': 100,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_GTSRB_BadNets'
    }
    finetuning.repair(schedule)
    repair_model = finetuning.get_model()
    torch.save(repair_model, save_path)
    
    del finetuning

train_dataset = SpeechCommandsDataset(param.path.benign_train_npypath)
test_dataset =  SpeechCommandsDataset(param.path.benign_test_npypath)

poisoned_test_dataset =  SpeechCommandsDataset('./datasets/temp/test_bit/')


model = load_model(attack_model_name)
print(f'finetuning_epoch: {finetuning_epoch}')
test_finetuning(model, save_model_name, p, train_dataset, test_dataset, poisoned_test_dataset, layer, y_target)