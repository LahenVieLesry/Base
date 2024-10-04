'''
This is the test code of NAD defense.
'''


from copy import deepcopy
import os
import os.path as osp
from cv2 import transform

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from defense.ABL import ABL
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from datasets import *
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize

from param import param

# os.environ['CUDA_VISIBLE_DEVICES'] = f'{param.GPU_num}'

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
gamma = param.defense.ABL.gamma
pre_epoch = param.defense.ABL.pre_epoch
clean_epoch = param.defense.ABL.clean_epoch
unlearn_epoch = param.defense.ABL.unlearn_epoch

#choose the proportion of data to calculation
y_target = class_to_idx[param.trigger_gen.target_label]

# model_name = param.trigger_train.model_name
# _model_name = model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr) + '_' +'mom_'+str(momentum) + '_'  + 'id_' + param.trigger_train.id + '.pth'
_model_name = '000.pth'
attack_model_name = './models/train/' + _model_name
save_model_name = f'models/defense/abl_pre{pre_epoch}_clean{clean_epoch}_unlearn{unlearn_epoch}' + _model_name

def load_model(path):
    print('Loading a pretrained model ')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    return model

def test(defense, defend, model_name, dataset_name, attack_name, defense_name, split_ratio, isolation_criterion, gamma, transform, selection_criterion, pre_epoch, clean_epoch, unlearn_epoch):
    if defend:
        pre_epoch, clean_epoch, unlearn_epoch, gamma, exp_detail = pre_epoch, clean_epoch, unlearn_epoch, gamma, 'w-defense'
    else:
        pre_epoch, clean_epoch, unlearn_epoch, gamma, exp_detail = 100, 0, 0, 0, 'wo-defense'
    
    # 5 unlearning epoch for badnets is enough
    # 5 unlearning epoch for wanet is enough

    pre_isolation_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'epochs': pre_epoch, 
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'schedule': [],
        'gamma': 0.1,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
    }

    clean_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'epochs': clean_epoch, 
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'schedule': [30, 55],
        'gamma': 0.1,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
    }

    unlearning_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'epochs': unlearn_epoch, 
        'lr': 5e-4,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'schedule': [],
        'gamma': 0.1,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
    }

    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    
    split_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,
    }

    schedule = {
        'save_dir': 'experiments/ABL-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{exp_detail}'+'_%.3f'%split_ratio+'_%.3f'%gamma,

        'pre_isolation_schedule': pre_isolation_schedule,
        'split_schedule': split_schedule,
        'clean_schedule': clean_schedule,
        'unlearning_schedule': unlearning_schedule,
        'test_schedule': test_schedule,
    }

    defense.train(split_ratio=split_ratio,
                  isolation_criterion = isolation_criterion,
                  gamma = gamma,
                  schedule=schedule,
                  transform=transform,
                  selection_criterion=selection_criterion)

    return 


train_dataset = SpeechCommandsDataset(param.path.benign_train_npypath)
test_dataset =  SpeechCommandsDataset(param.path.benign_test_npypath)

poisoned_train_dataset =  SpeechCommandsDataset(param.path.poison_train_path)
poisoned_test_dataset =  SpeechCommandsDataset('datasets/temp/test_bit_c/')

# model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GSCv1', 'Wav', 'ABL'
model_name, dataset_name, attack_name, defense_name = 'resnet34', 'dataset_name', 'wav', 'ABL'
torch.manual_seed(global_seed)

model = load_model(attack_model_name)

defense = ABL(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    poisoned_testset=poisoned_test_dataset,
    clean_testset=test_dataset,
    seed=global_seed,
    deterministic=deterministic
)
test(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name,
     attack_name=attack_name,
     defense_name=defense_name, 
     split_ratio=0.01,
     isolation_criterion=nn.CrossEntropyLoss(reduction='none'),
     gamma=gamma,
     transform=None,
    #  transform=Compose(
    #      [transforms.RandomCrop(32, padding=4),
    #       transforms.RandomHorizontalFlip(),]
    #  ),
     selection_criterion=nn.CrossEntropyLoss(reduction='none'),
     pre_epoch=pre_epoch,
     clean_epoch=clean_epoch,
     unlearn_epoch=unlearn_epoch)
repair_model = defense.get_model()
torch.save(repair_model, save_model_name)