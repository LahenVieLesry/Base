'''
This is the test code of IBD-PSC defense.
IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 

'''


from copy import deepcopy
import os.path as osp

import numpy as np
import random
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from datasets import *
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize

from defense.IBD_PSC import IBD_PSC
from param import param
from log import log


# ========== Set global settings ==========
global_seed = 7
# deterministic = True
deterministic = False
torch.manual_seed(global_seed)
# ========== Set global settings ==========
# datasets_root_dir = os.path.expanduser('~/data/dataset')
CUDA_VISIBLE_DEVICES = f'cuda:{param.GPU_num}'
portion = 0.1
batch_size = param.trigger_train.batch_size
num_workers = param.trigger_train.num_workers
n = param.defense.IBD_PSC.n
xi = param.defense.IBD_PSC.xi
T = param.defense.IBD_PSC.T

_model_name = '000.pth'
benign_model_name = 'models/train/resnet34_epochs_200_batchsize_128_Adam_lr_0.02_mom_0.9_id_0.pth'
attack_model_name = './models/train/' + _model_name
save_model_name = f'models/defense/IBDPSC_' + _model_name

def load_model(path):
    print('Loading a pretrained model')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    return model

def test(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target):
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/IBD-PSC-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }
    # defense.test_acc(benign_dataset, schedule)
    if not attack_name == 'Benign':
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,

            'batch_size': batch_size,
            'num_workers': num_workers,

            # 1. ASR: the attack success rate calculated on all poisoned samples
            # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
            # 3. BA: the accuracy on all benign samples
            # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
            # In other words, ASR or BA does not influence the computation of the metric.
            # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
            'metric': 'ASR_NoTarget',
            'y_target': y_target,

            'save_dir': 'experiments/IBD-PSC-defense',
            'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR'
        }
        # defense.test_acc(attacked_dataset, schedule)

model_name, dataset_name, attack_name, defense_name = 'resnet34', 'CIFAR-10', 'Benign', 'IBD-PSC'

benign_model = load_model(benign_model_name)

transform_train = None
trainset = SpeechCommandsDataset(param.path.benign_train_npypath)
transform_test = None
testset = SpeechCommandsDataset(param.path.benign_test_npypath)
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]
val_set = Subset(testset, val_indices)

defense = IBD_PSC(model=benign_model, valset=val_set)
test(model_name, dataset_name, attack_name, defense_name, testset, None, defense, None)

model_name, dataset_name, attack_name, defense_name = 'resnet34', 'dataset_name', 'wav', 'IBD-PSC'

poi_model = load_model(attack_model_name)

poisoned_trainset = SpeechCommandsDataset(param.path.poison_train_path)
poisoned_testset = SpeechCommandsDataset('datasets/temp/test_bit_c/')
# poisoned_testset = Subset(poisoned_testset, test_indices)
defense = IBD_PSC(model=poi_model, valset=val_set, n=n, xi=xi, T=T)
# print(f'the BA and ASR of the original poisoned model: ............. ')
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, None)
TPR, FPR, auc, myf1 = defense.test(testset, poisoned_testset)

# `detect` function is used to check if the first batch of samples in the dataset is poisoned.
# Users can assemble their data into a batch of shape [num_samples, n, w, h] and call `defense._detect(batch)`
# for online detection of the input.
# `preds_benign` contains the detection results for the original test dataset.
# `preds_poison` contains the detection results for the poisoned test dataset.
preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
# print(f'Is poisoned for real benign batch: {preds_benign}')
# print(f'Is poisoned for real poisoned batch: {preds_poison}')
log(f'IBD-PSC', f'\nn:{n}\nxi:{xi}\nT:{T}' + "\nTPR: {:.2f}\n".format(TPR) + "FPR: {:.2f}\n".format(FPR) + "AUC: {:.4f}\n".format(auc) + f"f1 score: {myf1}\n" + f'Is poisoned for real benign batch:\n{preds_benign}\n' + f'Is poisoned for real poisoned batch:\n{preds_poison}\n')
