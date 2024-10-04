'''
This is the test code of Spectral defense.
'''
import os
from copy import deepcopy
import os.path as osp
from cv2 import transform

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize

from defense.Spectral import Spectral
from param import param
from datasets import *


# ========== Set global settings ==========
global_seed = 7
deterministic = True
torch.manual_seed(global_seed)

datasets_root_dir = '../dataset/'
CUDA_VISIBLE_DEVICES = f'{param.GPU_num}'
batch_size = param.trigger_train.batch_size
num_workers = param.trigger_train.num_workers

_model_name = '000.pth'
benign_model_name = 'models/train/resnet34_epochs_200_batchsize_128_Adam_lr_0.02_mom_0.9_id_0.pth'
attack_model_name = './models/train/' + _model_name
save_model_name_benign = f'models/defense/NAD_benign_' + _model_name
save_model_name_poi = f'models/defense/NAD_poi_' + _model_name

def load_model(path):
    print('Loading a pretrained model')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    return model

model_name, dataset_name, attack_name, defense_name = 'resnet34', 'dataset_name', 'Benign', 'NAD'

transform_train = None
trainset = SpeechCommandsDataset(param.path.benign_train_npypath)
transform_test = None
testset = SpeechCommandsDataset(param.path.benign_test_npypath)

# ===================== BadNets ======================

torch.manual_seed(global_seed)
model = load_model(benign_model_name)

poisoned_train_dataset = SpeechCommandsDataset(param.path.poison_train_path)
poisoned_testset = SpeechCommandsDataset('datasets/temp/test_bit_c/')

# defend against BadNets attack
defense = Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': f'{param.GPU_num}',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'spectral_badnets_cifar10'
}

defense.test(poisoned_location = poisoned_train_dataset.poison_set(), schedule = schedule)